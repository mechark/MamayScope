#!/usr/bin/env python3
"""
Push a datasets.save_to_disk() cached-activations dataset to the Hugging Face Hub.

Prereqs:
  - `huggingface-cli login` OR set `HF_TOKEN` in env.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_dir",
        required=True,
        help="Path to dataset directory created by datasets.save_to_disk().",
    )
    p.add_argument(
        "--repo_id",
        required=True,
        help="Hugging Face dataset repo_id (e.g. username/dataset-name).",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create/push as a private dataset repo.",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Optional number of shards when uploading (leave unset to let HF decide).",
    )
    p.add_argument(
        "--revision",
        default="main",
        help="Target revision/branch (default: main).",
    )
    p.add_argument(
        "--commit_message",
        default="Add SAELens cached activations dataset",
        help="Commit message for the hub upload.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ds_path = Path(args.dataset_dir)
    if not ds_path.exists():
        raise FileNotFoundError(f"dataset_dir does not exist: {ds_path}")

    ds = load_from_disk(str(ds_path))
    print("Loaded dataset")
    print(ds)
    print("Features:", ds.features)

    ds.push_to_hub(
        repo_id=args.repo_id,
        num_shards=args.num_shards,
        private=bool(args.private),
        revision=args.revision,
        commit_message=args.commit_message,
    )
    print(f"Pushed to hub: {args.repo_id}")


if __name__ == "__main__":
    main()

