#!/usr/bin/env python3
"""
Build SAE feature occurrence -> context window examples from neuron-label Parquet batches.

Each Parquet row is a source text with a nested `tokens` list. Each token has `fired_features`
(SAELens sparse feature indices). We reverse the mapping:

  Sentence -> Token -> Features
  becomes
  Feature -> sampled list of occurrences, where each occurrence stores a broader
  context window (default: 20 tokens before + fired token + 20 tokens after).

This script can run in index-only mode via --skip-llm.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq

from src.services.openrouter_labeling_service import OpenRouterLabelingService


_LOW_INFO_EXACT_TOKENS = {
    "",
    "<PAD>",
    "<bos>",
    "<eos>",
    "<unk>",
    "[]",
    "()",
}

_LOW_INFO_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
    # Common Ukrainian stop words.
    "а",
    "але",
    "бо",
    "в",
    "від",
    "вона",
    "вони",
    "воно",
    "він",
    "до",
    "з",
    "за",
    "й",
    "і",
    "їх",
    "його",
    "ми",
    "на",
    "не",
    "про",
    "та",
    "ти",
    "то",
    "у",
    "це",
    "ця",
    "ці",
    "цей",
    "що",
    "я",
}

_GENERIC_LABEL_PHRASES = (
    "key concept",
    "concept identification",
    "semantic element",
    "semantic entities",
    "noun/descriptor",
    "categorical and relational",
    "relational or comparative",
)


@dataclass(frozen=True)
class OccurrenceContext:
    context: str
    token: str
    token_id: int

    def to_prompt_line(self) -> str:
        # Put context on its own line; LLM can read the brackets around the fired token.
        return self.context


def _normalize_token_for_prompt(token: str) -> str:
    t = str(token)
    # Avoid confusing JSON prompt formatting; keep the readability.
    t = t.replace("[", "(").replace("]", ")")
    return t.strip()


def _normalize_token_for_context(token: str) -> str:
    """Keep token spacing for context reconstruction, only sanitize control chars/brackets."""
    t = str(token)
    t = t.replace("[", "(").replace("]", ")")
    t = t.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return t


def _highlight_token_for_context(token: str) -> str:
    """
    Wrap a fired token with brackets while preserving leading whitespace.
    This keeps subword-detokenized text readable in multilingual contexts.
    """
    leading_ws_len = len(token) - len(token.lstrip(" "))
    leading_ws = token[:leading_ws_len]
    core = token[leading_ws_len:]
    core = core if core else "<PAD>"
    return f"{leading_ws}[{core}]"


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str) -> Any | None:
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as exc:
        print(f"[warn] Failed to load tokenizer for {model_name}: {exc}")
        return None


def _is_low_information_token(token: str) -> bool:
    """Reject structural-only fired tokens that dominate generic punctuation labels."""
    t = _normalize_token_for_prompt(token)
    t_lower = t.lower()
    if t in _LOW_INFO_EXACT_TOKENS:
        return True
    if t_lower in _LOW_INFO_STOPWORDS:
        return True
    # Ignore punctuation-only and delimiter fragments.
    if re.fullmatch(r"[^\w\s]+", t):
        return True
    # One-char non-alnum markers are usually structural noise for labeling.
    if len(t) == 1 and not t.isalnum():
        return True
    return False


def _token_diversity_stats(occurrences: list[OccurrenceContext]) -> tuple[int, float]:
    """Return unique token count and unique/total ratio for sampled contexts."""
    if not occurrences:
        return 0, 0.0
    normalized_tokens = [
        _normalize_token_for_prompt(occ.token).lower() for occ in occurrences if occ.token
    ]
    if not normalized_tokens:
        return 0, 0.0
    unique_count = len(set(normalized_tokens))
    return unique_count, unique_count / max(1, len(normalized_tokens))


def _token_entropy(occurrences: list[OccurrenceContext]) -> float:
    normalized_tokens = [
        _normalize_token_for_prompt(occ.token).lower() for occ in occurrences if occ.token
    ]
    total = len(normalized_tokens)
    if total <= 1:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for tok in normalized_tokens:
        counts[tok] += 1
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def order_features_for_labeling(
    feature_counts: dict[int, int],
    sampled_occurrences: dict[int, list[OccurrenceContext]],
) -> list[int]:
    """
    Rank features for labeling by token diversity first, then support.

    Prioritizing diverse fired-token samples tends to produce more semantic labels
    and de-prioritizes generic grammar/function-word features.
    """

    def _priority(fid: int) -> tuple[float, int, int, int]:
        occs = sampled_occurrences.get(fid, [])
        unique_count, diversity_ratio = _token_diversity_stats(occs)
        return (
            diversity_ratio,
            unique_count,
            len(occs),
            feature_counts.get(fid, 0),
        )

    return sorted(sampled_occurrences.keys(), key=_priority, reverse=True)


def apply_feature_quality_gate(
    ordered_feature_ids: list[int],
    sampled_occurrences: dict[int, list[OccurrenceContext]],
    *,
    min_unique_tokens: int,
    min_diversity_ratio: float,
    min_token_entropy: float,
) -> list[int]:
    gated: list[int] = []
    for fid in ordered_feature_ids:
        occs = sampled_occurrences.get(fid, [])
        unique_count, diversity_ratio = _token_diversity_stats(occs)
        entropy = _token_entropy(occs)
        if unique_count < min_unique_tokens:
            continue
        if diversity_ratio < min_diversity_ratio:
            continue
        if entropy < min_token_entropy:
            continue
        gated.append(fid)
    return gated


def is_generic_label(label: str) -> bool:
    norm = " ".join(str(label or "").strip().lower().split())
    if not norm:
        return True
    return any(phrase in norm for phrase in _GENERIC_LABEL_PHRASES)


def _normalize_label_key(label: str) -> str:
    return " ".join(str(label or "").strip().lower().split())


def build_context_window(
    tokens: list[dict[str, Any]],
    fired_token_index: int,
    *,
    radius: int = 20,
    tokenizer: Any | None = None,
) -> tuple[str, str, int]:
    """
    Build a fixed window around `fired_token_index`.

    We include `radius` tokens before + 1 fired token + `radius` tokens after.
    If indices go out of bounds, use '<PAD>'.
    """
    left = fired_token_index - radius
    right = fired_token_index + radius

    fired = tokens[fired_token_index]
    fired_token_str = _normalize_token_for_prompt(fired.get("token_str", ""))
    fired_token_id = int(fired.get("token_id", -1))

    window_parts: list[str] = []
    left_pad_count = 0
    right_pad_count = 0
    for i in range(left, right + 1):
        if i < 0 or i >= len(tokens):
            window_parts.append("<PAD>")
            if i < 0:
                left_pad_count += 1
            else:
                right_pad_count += 1
            continue
        tok = tokens[i]
        tok_str = _normalize_token_for_context(tok.get("token_str", ""))
        if i == fired_token_index:
            window_parts.append(_highlight_token_for_context(tok_str))
        else:
            window_parts.append(tok_str)

    if tokenizer is not None and fired_token_id >= 0:
        try:
            left_ids: list[int] = []
            right_ids: list[int] = []
            for i in range(left, right + 1):
                if i < 0 or i >= len(tokens):
                    continue
                tok_id = int(tokens[i].get("token_id", -1))
                if tok_id < 0:
                    continue
                if i < fired_token_index:
                    left_ids.append(tok_id)
                elif i > fired_token_index:
                    right_ids.append(tok_id)

            left_text = tokenizer.decode(
                left_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            fired_text = tokenizer.decode(
                [fired_token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            right_text = tokenizer.decode(
                right_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            core = (
                f"{left_text}"
                f"{_highlight_token_for_context(_normalize_token_for_context(fired_text))}"
                f"{right_text}"
            )
            prefix = " ".join(["<PAD>"] * left_pad_count)
            suffix = " ".join(["<PAD>"] * right_pad_count)
            context = " ".join(part for part in (prefix, core, suffix) if part)
        except Exception:
            context = " ".join(window_parts)
    else:
        context = " ".join(window_parts)

    context = re.sub(r"\s+([,.;:!?])", r"\1", context)
    context = re.sub(r"\s+", " ", context).strip()
    return context, fired_token_str, fired_token_id


def iter_parquet_rows(
    parquet_paths: list[str],
    *,
    columns: list[str],
) -> Iterator[dict[str, Any]]:
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=columns)
            n = table.num_rows
            # Materialize columns lazily per row.
            col_map = {name: table.column(name) for name in columns}
            for i in range(n):
                row: dict[str, Any] = {}
                for name in columns:
                    v = col_map[name][i].as_py()
                    row[name] = v
                yield row


def scan_feature_occurrences(
    *,
    parquet_glob: str,
    top_k: int,
    seed: int,
    min_occurrences: int,
    context_radius: int,
    max_rows: int | None = None,
) -> tuple[
    str | None,
    str | None,
    dict[int, int],
    dict[int, list[OccurrenceContext]],
    int,
    int,
]:
    """
    Returns:
      model_name (from Parquet 'model' column; may be None),
      sae_id (from Parquet 'sae_id' column; may be None),
      feature_counts: dict[feature_id -> total token-position occurrences]
      sampled_occurrences: dict[feature_id -> list of OccurrenceContext] with size<=top_k
    """
    paths = sorted(glob.glob(parquet_glob))
    if not paths:
        raise FileNotFoundError(f"No Parquet files matched glob: {parquet_glob}")

    feature_counts: dict[int, int] = defaultdict(int)
    sampled_occurrences: dict[int, list[OccurrenceContext]] = defaultdict(list)
    sampled_usable_counts: dict[int, int] = defaultdict(int)
    occurrences_filtered_out = 0

    rng = random.Random(seed)

    resolved_model: str | None = None
    resolved_sae_id: str | None = None
    tokenizer: Any | None = None
    tokenizer_model_name: str | None = None

    emitted_rows = 0
    columns = ["model", "sae_id", "tokens", "text"]
    for row in iter_parquet_rows(paths, columns=columns):
        if max_rows is not None and emitted_rows >= max_rows:
            break
        emitted_rows += 1

        if resolved_model is None:
            resolved_model = row.get("model")
        if resolved_sae_id is None:
            resolved_sae_id = row.get("sae_id")
        if (
            isinstance(resolved_model, str)
            and tokenizer is None
            and tokenizer_model_name != resolved_model
        ):
            tokenizer = _load_tokenizer(resolved_model)
            tokenizer_model_name = resolved_model

        tokens = row.get("tokens") or []
        if not isinstance(tokens, list) or not tokens:
            continue

        # Precompute context per token position once.
        for t_idx in range(len(tokens)):
            tok = tokens[t_idx] or {}
            fired_features = tok.get("fired_features") or []
            if not fired_features:
                continue

            context, fired_token_str, fired_token_id = build_context_window(
                tokens, t_idx, radius=context_radius, tokenizer=tokenizer
            )
            occ = OccurrenceContext(
                context=context,
                token=fired_token_str,
                token_id=fired_token_id,
            )
            keep_for_labeling = not _is_low_information_token(fired_token_str)
            if not keep_for_labeling:
                occurrences_filtered_out += 1

            # Update reservoir sampling for every fired feature at this token position.
            for feature_id in fired_features:
                fid = int(feature_id)
                feature_counts[fid] += 1
                if not keep_for_labeling:
                    continue
                sampled_usable_counts[fid] += 1
                n = sampled_usable_counts[fid]

                if n <= top_k:
                    sampled_occurrences[fid].append(occ)
                    continue

                # Uniform reservoir replacement.
                j = rng.randrange(n)  # [0, n-1]
                if j < top_k:
                    sampled_occurrences[fid][j] = occ

    # Filter to features that meet min_occurrences.
    eligible = {fid for fid, c in feature_counts.items() if c >= min_occurrences}
    filtered_counts = {fid: c for fid, c in feature_counts.items() if fid in eligible}
    filtered_samples = {
        fid: sampled_occurrences[fid]
        for fid in eligible
        if sampled_occurrences.get(fid)
    }
    features_with_no_usable_contexts = len(eligible) - len(filtered_samples)

    return (
        resolved_model,
        resolved_sae_id,
        filtered_counts,
        filtered_samples,
        occurrences_filtered_out,
        features_with_no_usable_contexts,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scan neuron-label Parquet batches and build feature->context samples."
    )
    p.add_argument(
        "--input-parquet-glob",
        default="data/neurons_labeling_propaganda/*_batch_*.parquet",
        help="Glob for input neuron-label Parquet batches.",
    )
    p.add_argument("--top-k", type=int, default=20, help="Sample up to K contexts per feature.")
    p.add_argument(
        "--min-occurrences",
        type=int,
        default=20,
        help="Only keep features with total occurrences >= this threshold.",
    )
    p.add_argument(
        "--context-radius",
        type=int,
        default=20,
        help="Number of tokens on each side of the fired token in the context window.",
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for deterministic reservoir sampling.")
    p.add_argument("--max-rows", type=int, default=None, help="Limit rows for a quick test.")
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="Index-only run: scan and write summary to stdout.",
    )
    p.add_argument(
        "--output-jsonl",
        default="data/neurons_labeling_propaganda/results/neuronpedia_feature_labels.jsonl",
        help="Output JSONL path (used only when LLM is enabled).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If output exists, skip already-labeled features.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Alias for --resume: skip already-labeled features if output exists.",
    )
    p.add_argument("--openrouter-model", default=None, help="OpenRouter model override.")
    p.add_argument(
        "--openrouter-base-url",
        default=None,
        help="OpenRouter API base (default from settings OPENROUTER_BASE_URL; use https://openrouter.ai/api/v1).",
    )
    p.add_argument(
        "--openrouter-provider-only",
        default=None,
        help="Comma-separated provider slugs for OpenRouter provider.only (overrides OPENROUTER_PROVIDER_ONLY).",
    )
    p.add_argument(
        "--openrouter-provider-order",
        default=None,
        help="Comma-separated provider slugs for OpenRouter provider.order (overrides OPENROUTER_PROVIDER_ORDER).",
    )
    p.add_argument(
        "--max-features-to-label",
        type=int,
        default=None,
        help="Optional cap on number of ranked features to label.",
    )
    p.add_argument(
        "--min-unique-tokens",
        type=int,
        default=3,
        help="Minimum number of unique fired tokens in sampled contexts.",
    )
    p.add_argument(
        "--min-diversity-ratio",
        type=float,
        default=0.4,
        help="Minimum unique-token ratio among sampled contexts.",
    )
    p.add_argument(
        "--min-token-entropy",
        type=float,
        default=0.9,
        help="Minimum entropy over fired-token distribution in sampled contexts.",
    )
    return p.parse_args()


def _optional_csv_providers(s: str | None) -> list[str] | None:
    if s is None or not str(s).strip():
        return None
    parts = [p.strip().lower() for p in str(s).split(",") if p.strip()]
    return parts or None


def main() -> None:
    args = _parse_args()

    (
        model_name,
        sae_id,
        feature_counts,
        sampled,
        occurrences_filtered_out,
        features_with_no_usable_contexts,
    ) = scan_feature_occurrences(
        parquet_glob=args.input_parquet_glob,
        top_k=args.top_k,
        seed=args.seed,
        min_occurrences=args.min_occurrences,
        context_radius=args.context_radius,
        max_rows=args.max_rows,
    )

    ordered_feature_ids = order_features_for_labeling(feature_counts, sampled)
    ordered_feature_ids = apply_feature_quality_gate(
        ordered_feature_ids,
        sampled,
        min_unique_tokens=max(1, int(args.min_unique_tokens)),
        min_diversity_ratio=max(0.0, float(args.min_diversity_ratio)),
        min_token_entropy=max(0.0, float(args.min_token_entropy)),
    )
    if args.max_features_to_label is not None and args.max_features_to_label > 0:
        ordered_feature_ids = ordered_feature_ids[: args.max_features_to_label]

    n_features = len(ordered_feature_ids)
    total_occ = sum(feature_counts.values())
    print(
        json.dumps(
            {
                "model": model_name,
                "sae_id": sae_id,
                "eligible_features": n_features,
                "sum_occurrences_over_eligible": total_occ,
                "top_k": args.top_k,
                "min_occurrences": args.min_occurrences,
                "occurrences_filtered_out": occurrences_filtered_out,
                "features_with_no_usable_contexts": features_with_no_usable_contexts,
                "features_ranked_for_labeling": n_features,
                "max_features_to_label": args.max_features_to_label,
                "min_unique_tokens": args.min_unique_tokens,
                "min_diversity_ratio": args.min_diversity_ratio,
                "min_token_entropy": args.min_token_entropy,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.skip_llm:
        return

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    should_skip_existing = bool(args.resume or args.skip_existing)

    labeled_features: set[int] = set()
    if output_path.exists() and should_skip_existing:
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fid = obj.get("feature_id")
                if fid is None:
                    continue
                try:
                    labeled_features.add(int(fid))
                except Exception:
                    continue
    elif output_path.exists() and not should_skip_existing:
        raise FileExistsError(
            f"Output file exists: {output_path}. Re-run with --resume to skip already-labeled features."
        )

    openrouter = OpenRouterLabelingService(
        model=args.openrouter_model,
        base_url=args.openrouter_base_url,
        provider_only=_optional_csv_providers(args.openrouter_provider_only),
        provider_order=_optional_csv_providers(args.openrouter_provider_order),
    )

    # Append one JSON object per feature.
    used_label_keys: set[str] = set()
    with output_path.open("a", encoding="utf-8") as out_f:
        for idx, feature_id in enumerate(ordered_feature_ids):
            if feature_id in labeled_features:
                continue

            contexts = [occ.context for occ in sampled[feature_id]]
            result = openrouter.label_feature(feature_id=feature_id, contexts=contexts)
            label_key = _normalize_label_key(result.label)
            needs_retry = is_generic_label(result.label) or (label_key in used_label_keys)
            if needs_retry:
                banned = list(used_label_keys) + [result.label]
                result = openrouter.label_feature(
                    feature_id=feature_id,
                    contexts=contexts,
                    avoid_labels=banned,
                    force_specific=True,
                )
                label_key = _normalize_label_key(result.label)
            used_label_keys.add(label_key)

            neuronpedia_feature_id: str | None = None
            if model_name and sae_id:
                neuronpedia_feature_id = f"{model_name}@{sae_id}:{feature_id}"

            obj = {
                "feature_id": feature_id,
                "model": model_name,
                "sae_id": sae_id,
                "neuronpedia_feature_id": neuronpedia_feature_id,
                "label": result.label,
                "thought_process": result.thought_process,
                "top_contexts": contexts,
                "top_k": args.top_k,
                "sampled_from_total": feature_counts[feature_id],
            }
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out_f.flush()

            if (idx + 1) % 10 == 0:
                print(f"[progress] labeled {idx + 1}/{n_features} features")


if __name__ == "__main__":
    main()

