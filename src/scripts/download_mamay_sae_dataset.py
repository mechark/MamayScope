#!/usr/bin/env python3
"""
Download the Mamay cached-activations dataset from the Hugging Face Hub and save it
as a local datasets.save_to_disk() directory.

This is intended for GPU pods where you want the dataset fully local (faster than streaming).

Auth:
  - set HF_TOKEN=... (preferred) or HUGGINGFACE_HUB_TOKEN=...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo_id",
        default="mechark/MamaySAEDataset",
        help="HF dataset repo_id (default: mechark/MamaySAEDataset).",
    )
    p.add_argument(
        "--data_dir",
        default="data",
        help="Subdirectory in the dataset repo containing parquet shards (default: data).",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Split to download (default: train).",
    )
    p.add_argument(
        "--out_dir",
        default="MamaySAEDataset",
        help="Output directory for datasets.save_to_disk() (default: ./MamaySAEDataset).",
    )
    p.add_argument(
        "--hook_name",
        default="blocks.33.hook_resid_post",
        help="Column name to sanity-check (default: blocks.33.hook_resid_post).",
    )
    return p.parse_args()


def _require_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing HF auth token. Set HF_TOKEN=... (preferred) or HUGGINGFACE_HUB_TOKEN=..."
        )
    # `datasets`/`huggingface_hub` will pick this up from env; we also return it for clarity.
    return token


def main() -> None:
    args = _parse_args()
    _require_token()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Non-streaming download: this will fetch/caches parquet locally.
    ds = load_dataset(
        args.repo_id,
        data_dir=args.data_dir,
        split=args.split,
        streaming=False,
    )

    ds.save_to_disk(str(out_dir))

    # Quick verification: schema + first row shape
    print("Downloaded dataset to:", out_dir.resolve())
    print(ds)
    print("Columns:", ds.column_names)
    if args.hook_name in ds.column_names:
        row0 = ds[0][args.hook_name]
        print("Verified hook column:", args.hook_name)
        print("First row shape:", len(row0), len(row0[0]) if row0 else None)
    else:
        print(
            "WARNING: hook column not found:",
            args.hook_name,
            "available:",
            ds.column_names,
        )


if __name__ == "__main__":
    main()

