#!/usr/bin/env python3
"""
Convert MamayScope Parquet activation dumps into a SAELens-compatible cached-activations dataset.

Input (Parquet):
  - columns: `output_tensor`
  - each cell: nested list with shape [seq_len][d_in] (float64 in Parquet)

Output (HuggingFace datasets `save_to_disk` directory):
  - column: hook_name (e.g. "blocks.33.hook_resid_post")
  - each row: fixed array of shape [context_size, d_in] (float32)

The script packs activations across rows into contiguous windows to avoid truncation.
Only the final window may be zero-padded.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Iterator

import numpy as np
import pyarrow.parquet as pq
from datasets import Array2D, Dataset, Features


@dataclass
class BuildStats:
    n_parquet_files: int = 0
    total_sequences_seen: int = 0
    total_real_tokens: int = 0
    n_full_windows: int = 0
    n_partial_window: int = 0
    padded_tokens: int = 0
    padded_rows: int = 0

    def padded_token_fraction(self) -> float:
        denom = self.total_real_tokens + self.padded_tokens
        return (self.padded_tokens / denom) if denom else 0.0

    def final_dataset_rows(self) -> int:
        return self.n_full_windows + self.padded_rows


def _iter_output_sequences(
    parquet_paths: list[str],
    *,
    column: str,
    d_in: int,
) -> Generator[np.ndarray, None, None]:
    """
    Yield (seq_len, d_in) float32 arrays from Parquet files, row-group by row-group.
    """
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=[column])
            col = table[column]
            # col is list<list<double>> where each row is a sequence
            for i in range(table.num_rows):
                seq = col[i].as_py()  # list[list[float]]
                arr = np.asarray(seq, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != d_in:
                    raise ValueError(
                        f"Unexpected sequence shape in {path} row_group={rg} row={i}: "
                        f"got {arr.shape}, expected (seq_len, {d_in})"
                    )
                yield arr


def _pack_into_windows(
    sequences: Iterable[np.ndarray],
    *,
    context_size: int,
    d_in: int,
    stats: BuildStats,
) -> Generator[np.ndarray, None, None]:
    """
    Pack token activations across sequences into fixed (context_size, d_in) windows.
    """
    carry = np.zeros((0, d_in), dtype=np.float32)

    for seq in sequences:
        stats.total_sequences_seen += 1
        stats.total_real_tokens += int(seq.shape[0])

        if carry.size == 0:
            buf = seq
        else:
            buf = np.concatenate([carry, seq], axis=0)

        # Emit full windows
        while buf.shape[0] >= context_size:
            window = buf[:context_size]
            yield window
            stats.n_full_windows += 1
            buf = buf[context_size:]

        carry = buf

    # Final partial window (pad once, at most)
    if carry.shape[0] > 0:
        stats.n_partial_window = 1
        padded = context_size - int(carry.shape[0])
        stats.padded_tokens = padded
        stats.padded_rows = 1
        out = np.zeros((context_size, d_in), dtype=np.float32)
        out[: carry.shape[0]] = carry
        yield out


def _maybe_filter_parquet_paths(parquet_paths: list[str], allow_non_batch: bool) -> list[str]:
    """
    Your `data/` directory contains some Parquet files that are not in the expected full-sequence
    2D format. The full-sequence batch files include `batch_` in the filename.
    """
    if allow_non_batch:
        return parquet_paths
    batch_paths = [p for p in parquet_paths if "batch_" in Path(p).name]
    if batch_paths:
        skipped = [p for p in parquet_paths if p not in batch_paths]
        if skipped:
            print(
                f"[filter] Skipping {len(skipped)} non-batch Parquet files. "
                f"Using {len(batch_paths)} batch files only."
            )
        return batch_paths
    return parquet_paths


def _consolidate_shard_dirs(shards_dir: Path, final_dataset_dir: Path) -> None:
    shard_dirs = sorted(
        [p for p in shards_dir.iterdir() if p.is_dir() and p.name.startswith("shard_")],
        key=lambda p: p.name,
    )
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* directories found under: {shards_dir}")

    first = shard_dirs[0]
    dataset_info_src = first / "dataset_info.json"
    state_src = first / "state.json"
    if not dataset_info_src.exists() or not state_src.exists():
        raise FileNotFoundError(
            f"Expected dataset_info.json and state.json under {first}."
        )

    final_dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_info_src, final_dataset_dir / "dataset_info.json")

    first_state = json.loads(state_src.read_text(encoding="utf-8"))

    # Collect all arrow files from all shards.
    arrow_files: list[tuple[int, Path]] = []
    for sd in shard_dirs:
        for f in sd.glob("data-*.arrow"):
            m = re.match(r"data-(\d+)-of-(\d+)\.arrow$", f.name)
            if not m:
                continue
            arrow_files.append((int(m.group(1)), f))

    if not arrow_files:
        raise FileNotFoundError(f"No data-*.arrow files found under: {shards_dir}")

    arrow_files.sort(key=lambda t: t[0])
    total = len(arrow_files)

    new_data_files: list[dict[str, str]] = []
    for idx, (_old_key, src_arrow) in enumerate(arrow_files):
        new_name = f"data-{idx:05d}-of-{total:05d}.arrow"
        dst_arrow = final_dataset_dir / new_name
        shutil.copy2(src_arrow, dst_arrow)
        new_data_files.append({"filename": new_name})

    new_state = dict(first_state)
    new_state["_data_files"] = new_data_files
    new_state["_fingerprint"] = None
    (final_dataset_dir / "state.json").write_text(
        json.dumps(new_state, indent=2), encoding="utf-8"
    )


def build_dataset(
    *,
    parquet_glob: str,
    hook_name: str,
    context_size: int,
    d_in: int,
    out_dir: Path,
    parquet_column: str = "output_tensor",
    shard_rows: int = 1000,
    writer_batch_size: int = 16,
    allow_non_batch: bool = False,
    log_every_rows: int = 100,
) -> BuildStats:
    parquet_paths = sorted(glob.glob(parquet_glob))
    if not parquet_paths:
        raise FileNotFoundError(f"No Parquet files matched glob: {parquet_glob}")
    parquet_paths = _maybe_filter_parquet_paths(
        parquet_paths, allow_non_batch=allow_non_batch
    )
    if not parquet_paths:
        raise FileNotFoundError(
            f"No Parquet files left after filtering for glob: {parquet_glob}"
        )

    stats = BuildStats(n_parquet_files=len(parquet_paths))

    features = Features(
        {
            hook_name: Array2D(shape=(context_size, d_in), dtype="float32"),
            # Optional token_ids could be added here if present in your data.
        }
    )

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}. Choose a new path or clear it first."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    shards_dir = out_dir / "_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    seqs = _iter_output_sequences(parquet_paths, column=parquet_column, d_in=d_in)
    window_iter = _pack_into_windows(
        seqs, context_size=context_size, d_in=d_in, stats=stats
    )

    rows_emitted_total = 0
    shard_idx = 0

    while True:
        shard_windows: list[np.ndarray] = []
        for _ in range(shard_rows):
            try:
                w = next(window_iter)
            except StopIteration:
                break
            shard_windows.append(w)

        if not shard_windows:
            break

        rows_emitted_total += len(shard_windows)
        if log_every_rows and (rows_emitted_total % log_every_rows == 0):
            print(
                f"[progress] rows={rows_emitted_total} real_tokens={stats.total_real_tokens} "
                f"padded_tokens={stats.padded_tokens}"
            )

        shard_path = shards_dir / f"shard_{shard_idx:05d}"
        shard_path.mkdir(parents=False, exist_ok=False)

        print(
            f"[shard] building {shard_path.name} rows_in_shard={len(shard_windows)}"
        )
        # Build shard from already-materialized windows to avoid `datasets.from_generator`
        # fingerprinting issues with stateful iterators.
        ds = Dataset.from_dict({hook_name: shard_windows}, features=features)
        ds.save_to_disk(str(shard_path))
        print(f"[shard] saved {shard_path}")
        shard_idx += 1

    # Final stats print (requested)
    padded_tokens = (context_size - (stats.total_real_tokens % context_size)) % context_size
    stats.padded_tokens = padded_tokens if stats.total_real_tokens else 0
    stats.padded_rows = 1 if stats.padded_tokens > 0 else 0
    stats.n_full_windows = stats.total_real_tokens // context_size if stats.total_real_tokens else 0
    stats.n_partial_window = 1 if stats.padded_rows else 0

    print("=== SAELens cached-activations build stats ===")
    print(f"context_size: {context_size}")
    print(f"d_in: {d_in}")
    print(f"hook_name: {hook_name}")
    print(f"parquet_glob: {parquet_glob}")
    print(f"parquet_column: {parquet_column}")
    print(f"n_parquet_files: {stats.n_parquet_files}")
    print(f"total_sequences_seen: {stats.total_sequences_seen}")
    print(f"total_real_tokens: {stats.total_real_tokens}")
    print(f"n_full_windows: {stats.n_full_windows}")
    print(f"n_partial_window: {stats.n_partial_window}")
    print(f"padded_tokens: {stats.padded_tokens}")
    print(f"padded_rows: {stats.padded_rows}")
    print(f"padded_token_fraction: {stats.padded_token_fraction():.8f}")
    print(f"final_dataset_rows: {stats.final_dataset_rows()}")
    print(f"shards_built: {shard_idx}")

    print("[merge] consolidating shards...")
    _consolidate_shard_dirs(shards_dir=shards_dir, final_dataset_dir=out_dir)
    print(f"saved_to: {out_dir}")

    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--parquet_glob",
        default="data/*batch_*.parquet",
        help="Glob for input Parquet files (default: data/*batch_*.parquet).",
    )
    p.add_argument(
        "--hook_name",
        default="blocks.33.hook_resid_post",
        help='Hook column name to use in the output dataset (default: "blocks.33.hook_resid_post").',
    )
    p.add_argument("--context_size", type=int, default=1024)
    p.add_argument("--d_in", type=int, default=3584)
    p.add_argument(
        "--out_dir",
        default="saelens_cached_activations_blocks33_resid_post",
        help="Output directory for datasets.save_to_disk().",
    )
    p.add_argument(
        "--parquet_column",
        default="output_tensor",
        help='Parquet column to read sequences from (default: "output_tensor").',
    )
    p.add_argument(
        "--shard_rows",
        type=int,
        default=256,
        help="Number of cached-activation rows (context_size x d_in) per shard.",
    )
    p.add_argument(
        "--writer_batch_size",
        type=int,
        default=16,
        help="(Currently unused) Kept for backwards compatibility.",
    )
    p.add_argument(
        "--allow_non_batch",
        action="store_true",
        help="If set, do not filter out non-batch Parquet files from --parquet_glob.",
    )
    p.add_argument(
        "--log_every_rows",
        type=int,
        default=100,
        help="Print progress every N emitted rows (0 disables).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    build_dataset(
        parquet_glob=args.parquet_glob,
        hook_name=args.hook_name,
        context_size=int(args.context_size),
        d_in=int(args.d_in),
        out_dir=Path(args.out_dir),
        parquet_column=args.parquet_column,
        shard_rows=int(args.shard_rows),
        writer_batch_size=int(args.writer_batch_size),
        allow_non_batch=bool(args.allow_non_batch),
        log_every_rows=int(args.log_every_rows),
    )


if __name__ == "__main__":
    main()

