#!/usr/bin/env python3
"""
Build local SAE-Vis HTML from neuron-label Parquet batches.

This script reuses MamayScope settings for:
- SAE snapshot source (HF repo/revision)
- base model id for tokenizer / HookedSAETransformer

It reads token ids from Parquet rows (`tokens[*].token_id`), builds a token tensor,
and renders feature-centric SAE-Vis HTML for selected feature ids.

Examples:
  # 1) Validate setup (no HTML render)
  uv run -m src.scripts.build_sae_vis_from_parquet --dry-run --feature-ids 700 --max-rows 128

  # 2) Render one dashboard html using feature ids from JSONL
  uv run -m src.scripts.build_sae_vis_from_parquet --max-features 8

  # 3) Render one file per feature
  uv run -m src.scripts.build_sae_vis_from_parquet --feature-ids 700,2569,2640 --save-per-feature
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from sae_lens import SAE, HookedSAETransformer
from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_storing_fns import SaeVisData

from src.core.settings import settings

LOGGER = logging.getLogger("build_sae_vis_from_parquet")


def _resolve_device(pref: str) -> str:
    p = (pref or "auto").strip().lower()
    if p == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return p


def _sae_snapshot_dir() -> Path:
    safe = settings.SAE_HF_REPO_ID.replace("/", "__")
    return Path(settings.SAE_SNAPSHOT_CACHE_DIR) / f"{safe}_{settings.SAE_HF_REVISION[:12]}"


def _load_sae(device: str) -> SAE:
    local_dir = _sae_snapshot_dir()
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (local_dir / "cfg.json").is_file():
        LOGGER.info(
            "Downloading SAE snapshot %s @ %s -> %s",
            settings.SAE_HF_REPO_ID,
            settings.SAE_HF_REVISION,
            local_dir,
        )
        snapshot_download(
            repo_id=settings.SAE_HF_REPO_ID,
            revision=settings.SAE_HF_REVISION,
            local_dir=str(local_dir),
        )
    else:
        LOGGER.info("Using cached SAE snapshot at %s", local_dir)

    try:
        sae = SAE.load_from_disk(str(local_dir), device=device)
    except Exception as exc:
        if device == "mps":
            LOGGER.warning("SAE.load_from_disk failed on mps (%s); retrying cpu", exc)
            sae = SAE.load_from_disk(str(local_dir), device="cpu")
        else:
            raise
    sae.eval()
    return sae


def _iter_parquet_token_ids(paths: list[str]) -> list[list[int]]:
    import pyarrow.parquet as pq

    out: list[list[int]] = []
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["tokens"])
            col = table.column("tokens")
            for i in range(table.num_rows):
                tok_rows = col[i].as_py() or []
                ids: list[int] = []
                for t in tok_rows:
                    tid = int((t or {}).get("token_id", -1))
                    if tid >= 0:
                        ids.append(tid)
                if ids:
                    out.append(ids)
    return out


def _build_token_tensor(
    seqs: list[list[int]],
    *,
    seq_len: int,
    max_rows: int | None,
    pad_id: int,
) -> torch.Tensor:
    rows = seqs[: max_rows if max_rows is not None else len(seqs)]
    if not rows:
        raise ValueError("No usable token-id rows found in Parquet input.")

    packed: list[list[int]] = []
    for r in rows:
        if len(r) >= seq_len:
            packed.append(r[:seq_len])
        else:
            packed.append(r + [pad_id] * (seq_len - len(r)))
    return torch.tensor(packed, dtype=torch.long)


def _parse_feature_ids_csv(raw: str | None) -> list[int]:
    if not raw or not raw.strip():
        return []
    vals: list[int] = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(int(p))
    return vals


def _read_feature_ids_jsonl(path: str | None, max_features: int | None) -> list[int]:
    if not path:
        return []
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(f"feature ids jsonl does not exist: {fpath}")

    ids: list[int] = []
    with fpath.open("r", encoding="utf-8") as f:
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
            ids.append(int(fid))
            if max_features is not None and len(ids) >= max_features:
                break
    return ids


def _dedupe_keep_order(items: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build local SAE-Vis from neuron-label Parquet.")
    p.add_argument(
        "--parquet-glob",
        default="data/neuron_labels_mamay/*_batch_*.parquet",
        help="Glob of neuron-label parquet batches.",
    )
    p.add_argument(
        "--feature-ids",
        default="",
        help="Comma-separated feature ids, e.g. 700,2569,2640",
    )
    p.add_argument(
        "--feature-ids-file",
        default="data/neuron_labels_mamay/results/neuronpedia_feature_labels.jsonl",
        help="JSONL file with `feature_id` field (optional source of ids).",
    )
    p.add_argument("--max-features", type=int, default=8, help="Max features to render.")
    p.add_argument("--max-rows", type=int, default=2048, help="Max parquet rows for token tensor.")
    p.add_argument("--seq-len", type=int, default=128, help="Token sequence length for visualization.")
    p.add_argument(
        "--output-html",
        default="data/neuron_labels_mamay/results/sae_vis_feature_dashboard.html",
        help="Output HTML file path for combined render.",
    )
    p.add_argument(
        "--output-dir",
        default="data/neuron_labels_mamay/results/sae_vis_features",
        help="Output dir for per-feature html files.",
    )
    p.add_argument(
        "--save-per-feature",
        action="store_true",
        help="Save one html per feature id into --output-dir.",
    )
    p.add_argument(
        "--feature-for-main-html",
        type=int,
        default=None,
        help="Feature id to show in --output-html. Defaults to first selected id.",
    )
    p.add_argument("--device", default=settings.SAE_DEVICE, help="auto/cuda/mps/cpu.")
    p.add_argument(
        "--tl-model-name",
        default="",
        help="Optional TransformerLens-compatible model id override (e.g. google/gemma-2-9b-it).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load inputs/model/sae and print summary without rendering html.",
    )
    return p.parse_args()


def _resolve_tl_model_name(raw_model_name: str, override: str | None) -> str:
    if override and override.strip():
        return override.strip()
    n = (raw_model_name or "").strip()
    lower = n.lower()
    # MamayLM Gemma IT models are tokenizer-compatible with Gemma 2 IT.
    if "mamaylm-gemma-2-9b-it" in lower:
        return "google/gemma-2-9b-it"
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()

    parquet_paths = sorted(glob.glob(args.parquet_glob))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files matched: {args.parquet_glob}")

    file_ids = _read_feature_ids_jsonl(args.feature_ids_file, args.max_features)
    cli_ids = _parse_feature_ids_csv(args.feature_ids)
    feature_ids = _dedupe_keep_order(cli_ids + file_ids)
    if args.max_features > 0:
        feature_ids = feature_ids[: args.max_features]
    if not feature_ids:
        raise ValueError("No feature ids selected. Provide --feature-ids or a valid --feature-ids-file.")

    token_rows = _iter_parquet_token_ids(parquet_paths)
    model_name = (settings.NEURON_LABEL_MODEL_NAME or "").strip()
    if not model_name:
        raise ValueError("NEURON_LABEL_MODEL_NAME is empty; set model id in .env/settings.")
    tl_model_name = _resolve_tl_model_name(model_name, args.tl_model_name)

    device = _resolve_device(args.device)
    model = HookedSAETransformer.from_pretrained(tl_model_name, device=device)
    sae = _load_sae(device)

    tokenizer = model.tokenizer
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError("Could not determine pad_token_id/eos_token_id from model tokenizer.")

    tokens = _build_token_tensor(
        token_rows,
        seq_len=max(1, int(args.seq_len)),
        max_rows=args.max_rows,
        pad_id=int(pad_id),
    ).to(model.cfg.device)

    summary = {
        "model_name": model_name,
        "transformerlens_model_name": tl_model_name,
        "sae_repo_id": settings.SAE_HF_REPO_ID,
        "sae_revision": settings.SAE_HF_REVISION,
        "device": device,
        "n_parquet_files": len(parquet_paths),
        "n_token_rows": len(token_rows),
        "tokens_shape": list(tokens.shape),
        "n_features_selected": len(feature_ids),
        "feature_ids": feature_ids,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.dry_run:
        return

    vis_data = SaeVisData.create(
        sae=sae,
        model=model,
        tokens=tokens,
        cfg=SaeVisConfig(features=feature_ids),
        verbose=True,
    )

    out_main = Path(args.output_html)
    out_main.parent.mkdir(parents=True, exist_ok=True)
    main_feature = args.feature_for_main_html if args.feature_for_main_html is not None else feature_ids[0]
    vis_data.save_feature_centric_vis(
        filename=str(out_main),
        feature=int(main_feature),
        verbose=True,
    )

    created = [str(out_main)]
    if args.save_per_feature:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for fid in feature_ids:
            path = out_dir / f"feature_{fid}.html"
            vis_data.save_feature_centric_vis(filename=str(path), feature=int(fid), verbose=False)
            created.append(str(path))

    print(
        json.dumps(
            {
                "status": "ok",
                "main_feature": int(main_feature),
                "output_files": created,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
