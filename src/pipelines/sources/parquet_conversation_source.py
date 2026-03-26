import logging
from pathlib import Path

import pyarrow.dataset as ds

from src.core.settings import settings
from src.pipelines.base import PipelineStep


class ParquetConversationBatchSource(PipelineStep):
    """Yields batches of concatenated conversation texts from local Parquet."""

    def __init__(
        self,
        parquet_path: str | Path | None = None,
        conversation_column: str | None = None,
        batch_size: int | None = None,
        max_rows: int | None = None,
        dedup_exact: bool | None = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.parquet_path = Path(parquet_path or settings.NEURON_LABEL_PROPAGANDA_PARQUET_SOURCE_PATH)
        self.conversation_column = (
            conversation_column or settings.NEURON_LABEL_PROPAGANDA_CONVERSATION_COLUMN
        )
        self.batch_size = batch_size or settings.NEURON_LABEL_PROPAGANDA_BATCH_SIZE
        self.max_rows = (
            max_rows
            if max_rows is not None
            else settings.NEURON_LABEL_PROPAGANDA_MAX_ROWS
        )
        self.dedup_exact = (
            dedup_exact
            if dedup_exact is not None
            else settings.NEURON_LABEL_PROPAGANDA_DEDUP_EXACT
        )

        self._batch_iter = None
        self._pending_batch = None
        self._pending_row_idx = 0
        self._emitted = 0
        self._seen_texts: set[str] = set()
        self._scanner_batch_rows = max(1024, min(65536, self.batch_size * 128))

    def _ensure_scan(self) -> None:
        if self._batch_iter is not None:
            return
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet path not found: {self.parquet_path}")
        dataset = ds.dataset(str(self.parquet_path), format="parquet")
        if self.conversation_column not in dataset.schema.names:
            raise KeyError(
                f"Column {self.conversation_column!r} not in Parquet; available: {dataset.schema.names}"
            )
        scanner = dataset.scanner(
            columns=[self.conversation_column],
            batch_size=self._scanner_batch_rows,
        )
        self._batch_iter = iter(scanner.to_batches())
        self.logger.info(
            "Opened conversation parquet at %s (column=%s, dedup_exact=%s)",
            self.parquet_path,
            self.conversation_column,
            self.dedup_exact,
        )

    def _concat_conversation(self, conversation_obj: object) -> str | None:
        if not isinstance(conversation_obj, list):
            return None
        parts: list[str] = []
        for item in conversation_obj:
            if not isinstance(item, dict):
                continue
            value = item.get("value")
            if value is None:
                continue
            text = str(value).strip()
            if text:
                parts.append(text)
        if not parts:
            return None
        return "\n".join(parts)

    async def run(self, data: dict) -> dict:
        del data
        self._ensure_scan()
        texts: list[str] = []
        exhausted = False

        while len(texts) < self.batch_size:
            if self.max_rows is not None and self._emitted >= self.max_rows:
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
            column = batch.column(self.conversation_column)
            while self._pending_row_idx < batch.num_rows and len(texts) < self.batch_size:
                if self.max_rows is not None and self._emitted >= self.max_rows:
                    exhausted = True
                    break

                row_idx = self._pending_row_idx
                self._pending_row_idx += 1
                row_value = column[row_idx].as_py()
                merged_text = self._concat_conversation(row_value)
                if not merged_text:
                    continue
                if self.dedup_exact and merged_text in self._seen_texts:
                    continue
                if self.dedup_exact:
                    self._seen_texts.add(merged_text)

                texts.append(merged_text)
                self._emitted += 1

            if self._pending_row_idx >= batch.num_rows:
                self._pending_batch = None

            if exhausted:
                break

        self.logger.info(
            "Parquet conversation source: yielded %s rows (emitted total %s%s)",
            len(texts),
            self._emitted,
            f"/{self.max_rows}" if self.max_rows is not None else "",
        )
        return {
            "texts": texts,
            "labels": [None] * len(texts),
            "done": exhausted or len(texts) == 0,
        }
