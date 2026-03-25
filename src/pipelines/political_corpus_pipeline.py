"""
Single-module political-topic corpus pipeline: Parquet source → keyword filter → text Parquet sink.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import unicodedata
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from src.pipelines.base import PipelineStep
from src.pipelines.pipeline_executor import PipelineExecutor

__all__ = [
    "ParquetTextBatchSource",
    "UkrainianPoliticalKeywordFilter",
    "TextCorpusParquetSink",
    "load_keywords_from_yaml",
]

MIN_KEYWORD_LEN = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_KEYWORDS = REPO_ROOT / "configs" / "uk_political_corpus_keywords.yaml"


def _norm_keyword(k: str) -> str:
    return unicodedata.normalize("NFKC", k.strip().casefold())


def load_keywords_from_yaml(path: str | Path) -> list[str]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "keywords" in data:
        raw = data["keywords"]
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError(f"Expected 'keywords' list or top-level list in {path}")
    seen: set[str] = set()
    for item in raw:
        if item is None:
            continue
        kn = _norm_keyword(str(item))
        if len(kn) < MIN_KEYWORD_LEN:
            continue
        seen.add(kn)
    return sorted(seen, key=len, reverse=True)


class ParquetTextBatchSource(PipelineStep):
    """Yields batches of text strings from a local Parquet file or directory of Parquet files."""

    def __init__(
        self,
        parquet_path: str | Path,
        text_column: str = "text",
        batch_size: int = 32,
        max_rows: int | None = None,
        skip_rows: int = 0,
        language_column: str | None = None,
        language_filter: str | None = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.ParquetTextBatchSource")
        self.parquet_path = Path(parquet_path)
        self.text_column = text_column
        self.batch_size = batch_size
        self.max_rows = max_rows
        self.skip_rows = skip_rows
        self.language_column = language_column
        self.language_filter = language_filter

        self._batch_iter = None
        self._pending_batch = None
        self._pending_row_idx = 0
        self._skip_remaining = skip_rows
        self._emitted_total = 0
        self._scanner_batch_rows = max(1024, min(65536, batch_size * 128))

    def _ensure_scan(self):
        if self._batch_iter is not None:
            return
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet path not found: {self.parquet_path}")
        cols = [self.text_column]
        if self.language_column and self.language_filter:
            cols.append(self.language_column)
        cols = list(dict.fromkeys(cols))
        dataset = ds.dataset(str(self.parquet_path), format="parquet")
        scanner = dataset.scanner(columns=cols, batch_size=self._scanner_batch_rows)
        self._batch_iter = iter(scanner.to_batches())
        self.logger.info(
            "Opened Parquet dataset at %s (columns=%s)",
            self.parquet_path,
            cols,
        )

    def _row_text_and_lang(self, batch, row_idx: int) -> tuple[str | None, str | None]:
        tcol = batch.column(self.text_column)
        raw = tcol[row_idx]
        if raw is None:
            return None, None
        text = raw.as_py()
        if not isinstance(text, str):
            text = str(text) if text is not None else None
        if not text or not text.strip():
            return None, None
        lang = None
        if self.language_column and self.language_filter:
            lcol = batch.column(self.language_column)
            lraw = lcol[row_idx]
            if lraw is not None:
                lang = lraw.as_py()
                if lang is not None and not isinstance(lang, str):
                    lang = str(lang)
        return text, lang

    async def run(self, data: dict) -> dict:
        self._ensure_scan()
        texts: list[str] = []
        exhausted = False

        while len(texts) < self.batch_size:
            if self.max_rows is not None and self._emitted_total >= self.max_rows:
                exhausted = True
                break

            if self._pending_batch is None:
                try:
                    self._pending_batch = next(self._batch_iter)
                except StopIteration:
                    exhausted = True
                    break
                self._pending_row_idx = 0

            batch = self._pending_batch
            while self._pending_row_idx < batch.num_rows:
                if self.max_rows is not None and self._emitted_total >= self.max_rows:
                    exhausted = True
                    break
                row_idx = self._pending_row_idx
                self._pending_row_idx += 1
                text, lang = self._row_text_and_lang(batch, row_idx)
                if text is None:
                    continue
                if self.language_filter and self.language_column:
                    if lang is None or lang.strip().lower() != self.language_filter.strip().lower():
                        continue
                if self._skip_remaining > 0:
                    self._skip_remaining -= 1
                    continue
                texts.append(text)
                self._emitted_total += 1
                if len(texts) >= self.batch_size:
                    break
                if self.max_rows is not None and self._emitted_total >= self.max_rows:
                    exhausted = True
                    break

            if self._pending_row_idx >= batch.num_rows:
                self._pending_batch = None

            if exhausted:
                break

        done = exhausted
        self.logger.info(
            "Parquet source: yielded %s texts (total emitted: %s%s)",
            len(texts),
            self._emitted_total,
            f"/{self.max_rows}" if self.max_rows is not None else "",
        )

        return {
            "texts": texts,
            "done": done or len(texts) == 0,
        }


class UkrainianPoliticalKeywordFilter(PipelineStep):
    """Keeps texts that contain at least one Ukrainian political-topic keyword (substring match)."""

    def __init__(self, keywords: list[str] | None = None, keywords_yaml: str | Path | None = None):
        self.logger = logging.getLogger(f"{__name__}.UkrainianPoliticalKeywordFilter")
        if keywords is not None:
            seen: set[str] = set()
            for k in keywords:
                if not k:
                    continue
                kn = _norm_keyword(str(k))
                if len(kn) >= MIN_KEYWORD_LEN:
                    seen.add(kn)
            self._keywords = sorted(seen, key=len, reverse=True)
        elif keywords_yaml is not None:
            self._keywords = load_keywords_from_yaml(keywords_yaml)
        else:
            raise ValueError("Either keywords or keywords_yaml must be set")
        self._seen_in = 0
        self._kept = 0

    def _normalize(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text.casefold())

    def _matches(self, normalized: str) -> bool:
        return any(kw in normalized for kw in self._keywords)

    async def run(self, data: dict) -> dict:
        texts = data.get("texts", [])
        if not texts:
            self.logger.warning("Keyword filter: no texts in batch")
            return {
                "texts": [],
                "done": data.get("done", True),
            }

        kept: list[str] = []
        for t in texts:
            self._seen_in += 1
            if self._matches(self._normalize(t)):
                kept.append(t)
                self._kept += 1

        self.logger.info(
            "Keyword filter: kept %s / %s in batch (totals: kept=%s, seen=%s)",
            len(kept),
            len(texts),
            self._kept,
            self._seen_in,
        )

        return {
            "texts": kept,
            "done": data.get("done", False),
        }


class TextCorpusParquetSink(PipelineStep):
    """Appends filtered text rows to sharded Parquet files (one shard per pipeline batch with rows)."""

    def __init__(self, output_parquet_path: str | Path):
        self.logger = logging.getLogger(f"{__name__}.TextCorpusParquetSink")
        self.output_path = Path(output_parquet_path)
        self.batch_counter = 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = list(self.output_path.parent.glob(f"{self.output_path.stem}_batch_*.parquet"))
        if existing:
            self.batch_counter = max(int(f.stem.split("_batch_")[1]) for f in existing) + 1
            self.logger.info(
                "Found %s existing corpus batch files, next batch index %s",
                len(existing),
                self.batch_counter,
            )

    async def run(self, data: dict) -> dict:
        texts = data.get("texts", [])
        if not texts:
            return data

        df = pd.DataFrame({"text": texts})
        table = pa.Table.from_pandas(df)
        batch_path = (
            self.output_path.parent / f"{self.output_path.stem}_batch_{self.batch_counter:06d}.parquet"
        )
        pq.write_table(table, batch_path, compression="snappy")
        self.logger.info("Wrote %s texts to %s", len(texts), batch_path.name)
        self.batch_counter += 1
        del df, table
        return data


async def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Parquet texts by Ukrainian political keywords.")
    parser.add_argument("--input-parquet", required=True, type=Path, help="Parquet file or directory")
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/uk_political_corpus.parquet"),
        help="Output path stem (batches: <stem>_batch_######.parquet)",
    )
    parser.add_argument(
        "--keywords-yaml",
        type=Path,
        default=DEFAULT_KEYWORDS,
        help="YAML file with a `keywords:` list",
    )
    parser.add_argument("--text-column", default="text")
    batch_default = 512
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_default,
        help="Pipeline batch size (texts per pull from Parquet)",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Max source rows to emit (after skip)")
    parser.add_argument("--skip-rows", type=int, default=0, help="Skip this many qualifying source rows")
    parser.add_argument("--language-column", default="language", help="Column name when filtering by language")
    parser.add_argument(
        "--language",
        default=None,
        help="If set (e.g. uk), keep only rows where language column matches",
    )
    args = parser.parse_args()

    kw_path = Path(args.keywords_yaml)
    if not kw_path.is_file():
        raise FileNotFoundError(f"Keywords file not found: {kw_path}")

    lang_col = args.language_column if args.language else None
    lang_f = args.language.strip() if args.language else None

    logger.info("Starting political-topic corpus pipeline")
    logger.info("Input: %s", args.input_parquet)
    logger.info("Output stem: %s", args.output_parquet)
    logger.info("Keywords: %s", kw_path)

    executor = PipelineExecutor(
        steps=[
            ParquetTextBatchSource(
                parquet_path=args.input_parquet,
                text_column=args.text_column,
                batch_size=args.batch_size,
                max_rows=args.max_rows,
                skip_rows=args.skip_rows,
                language_column=lang_col,
                language_filter=lang_f,
            ),
            UkrainianPoliticalKeywordFilter(keywords_yaml=kw_path),
            TextCorpusParquetSink(output_parquet_path=args.output_parquet),
        ]
    )

    result: dict = {}
    iteration = 0
    while not result.get("done", False):
        iteration += 1
        logger.info("=== Pipeline iteration %s ===", iteration)
        result = await executor.execute({})
        if result.get("done"):
            logger.info("Pipeline completed.")
            break

    logger.info("Finished after %s iterations", iteration)


if __name__ == "__main__":
    asyncio.run(main())
