#!/usr/bin/env python3
"""
Build SAE feature occurrence -> context window examples from neuron-label Parquet batches.

Each Parquet row is a source text with a nested `tokens` list. Each token has `fired_features`
(SAELens sparse feature indices). We reverse the mapping:

  Sentence -> Token -> Features
  becomes
  Feature -> sampled list of occurrences, where each occurrence stores a 10ish-token
  context window (5 tokens before + fired token + 5 tokens after).

This script can run in index-only mode via --skip-llm.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq

from src.services.openrouter_labeling_service import OpenRouterLabelingService


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


def build_context_window(
    tokens: list[dict[str, Any]],
    fired_token_index: int,
    *,
    radius: int = 5,
) -> tuple[str, str, int]:
    """
    Build a fixed window around `fired_token_index`.

    We include 5 tokens before + 1 fired token + 5 tokens after => up to 11 tokens.
    If indices go out of bounds, use '<PAD>'.
    """
    left = fired_token_index - radius
    right = fired_token_index + radius

    fired = tokens[fired_token_index]
    fired_token_str = _normalize_token_for_prompt(fired.get("token_str", ""))
    fired_token_id = int(fired.get("token_id", -1))

    window_parts: list[str] = []
    for i in range(left, right + 1):
        if i < 0 or i >= len(tokens):
            window_parts.append("<PAD>")
            continue
        tok = tokens[i]
        tok_str = _normalize_token_for_prompt(tok.get("token_str", ""))
        if i == fired_token_index:
            window_parts.append(f"[{tok_str}]")
        else:
            window_parts.append(tok_str)

    context = " ".join(window_parts).strip()
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

    rng = random.Random(seed)

    resolved_model: str | None = None
    resolved_sae_id: str | None = None

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
                tokens, t_idx, radius=context_radius
            )
            occ = OccurrenceContext(
                context=context,
                token=fired_token_str,
                token_id=fired_token_id,
            )

            # Update reservoir sampling for every fired feature at this token position.
            for feature_id in fired_features:
                fid = int(feature_id)
                feature_counts[fid] += 1
                n = feature_counts[fid]

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
    filtered_samples = {fid: sampled_occurrences[fid] for fid in eligible}

    return resolved_model, resolved_sae_id, filtered_counts, filtered_samples


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scan neuron-label Parquet batches and build feature->context samples."
    )
    p.add_argument(
        "--input-parquet-glob",
        default="data/neuron_labels/*_batch_*.parquet",
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
        default=5,
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
        default="data/neuron_labels/neuronpedia_feature_labels.jsonl",
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
    return p.parse_args()


def _optional_csv_providers(s: str | None) -> list[str] | None:
    if s is None or not str(s).strip():
        return None
    parts = [p.strip().lower() for p in str(s).split(",") if p.strip()]
    return parts or None


def main() -> None:
    args = _parse_args()

    model_name, sae_id, feature_counts, sampled = scan_feature_occurrences(
        parquet_glob=args.input_parquet_glob,
        top_k=args.top_k,
        seed=args.seed,
        min_occurrences=args.min_occurrences,
        context_radius=args.context_radius,
        max_rows=args.max_rows,
    )

    n_features = len(feature_counts)
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
    with output_path.open("a", encoding="utf-8") as out_f:
        for idx, feature_id in enumerate(sorted(feature_counts.keys())):
            if feature_id in labeled_features:
                continue

            contexts = [occ.context for occ in sampled[feature_id]]
            result = openrouter.label_feature(feature_id=feature_id, contexts=contexts)

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

