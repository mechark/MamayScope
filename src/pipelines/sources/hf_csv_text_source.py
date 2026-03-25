import logging
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from src.core.settings import settings
from src.pipelines.base import PipelineStep


class HFCsvBatchSource(PipelineStep):
    """Yields batches of texts (and optional labels) from a CSV file on the Hugging Face Hub."""

    def __init__(
        self,
        repo_id: str | None = None,
        csv_filename: str | None = None,
        text_column: str | None = None,
        label_column: str | None = None,
        batch_size: int | None = None,
        max_rows: int | None = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.repo_id = repo_id or settings.NEURON_LABEL_HF_DATASET
        self.csv_filename = csv_filename or settings.NEURON_LABEL_CSV_FILE
        self.text_column = text_column or settings.NEURON_LABEL_TEXT_COLUMN
        self.label_column = label_column or settings.NEURON_LABEL_LABEL_COLUMN
        self.batch_size = batch_size or settings.BATCH_SIZE
        self.max_rows = max_rows if max_rows is not None else settings.NEURON_LABEL_DATASET_LIMIT

        self._df: pd.DataFrame | None = None
        self._cursor = 0
        self._emitted = 0

    def _ensure_loaded(self) -> None:
        if self._df is not None:
            return
        path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.csv_filename,
            repo_type="dataset",
        )
        self.logger.info("Loading CSV from %s (%s)", self.repo_id, path)
        self._df = pd.read_csv(
            path,
            on_bad_lines="skip",
            engine="python",
        )
        if self.text_column not in self._df.columns:
            raise KeyError(
                f"Column {self.text_column!r} not in CSV; available: {list(self._df.columns)}"
            )
        if self.label_column not in self._df.columns:
            self.logger.warning(
                "Label column %r not in CSV; downstream label will be null",
                self.label_column,
            )

    def _row_label(self, row: Any) -> Any:
        if self._df is None or self.label_column not in self._df.columns:
            return None
        raw = row[self.label_column]
        if pd.isna(raw):
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw

    async def run(self, data: dict) -> dict:
        del data  # source ignores upstream
        self._ensure_loaded()
        assert self._df is not None

        texts: list[str] = []
        labels: list[Any] = []
        exhausted = False

        while len(texts) < self.batch_size:
            if self.max_rows is not None and self._emitted >= self.max_rows:
                exhausted = True
                break
            if self._cursor >= len(self._df):
                exhausted = True
                break

            row = self._df.iloc[self._cursor]
            self._cursor += 1
            raw_text = row[self.text_column]
            if pd.isna(raw_text):
                continue
            text = str(raw_text).strip()
            if not text:
                continue

            texts.append(text)
            labels.append(self._row_label(row))
            self._emitted += 1

            if self.max_rows is not None and self._emitted >= self.max_rows:
                exhausted = True
                break

        done = exhausted or (len(texts) == 0 and self._cursor >= len(self._df))
        if len(texts) == 0 and self._cursor >= len(self._df):
            done = True

        self.logger.info(
            "HF CSV source: yielded %s rows (emitted total %s%s)",
            len(texts),
            self._emitted,
            f"/{self.max_rows}" if self.max_rows is not None else "",
        )

        return {
            "texts": texts,
            "labels": labels,
            "done": done,
        }
