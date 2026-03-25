import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.core.settings import settings
from src.pipelines.base import PipelineStep


class ParquetNeuronActivationSink(PipelineStep):
    """Writes neuron labeling records to sharded Parquet (one row per source text)."""

    _TOKEN_LIST_TYPE = pa.list_(
        pa.struct(
            [
                ("token_str", pa.string()),
                ("token_id", pa.int64()),
                ("fired_features", pa.list_(pa.int64())),
            ]
        )
    )

    def __init__(self, output_path: str | Path | None = None):
        self.logger = logging.getLogger(__name__)
        self.output_path = Path(output_path or settings.NEURON_LABEL_PARQUET_PATH)
        self.batch_counter = 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = list(self.output_path.parent.glob(f"{self.output_path.stem}_batch_*.parquet"))
        if existing:
            self.batch_counter = max(int(f.stem.split("_batch_")[1]) for f in existing) + 1
            self.logger.info(
                "Found %s existing neuron-label Parquet batches, next index %s",
                len(existing),
                self.batch_counter,
            )

    async def run(self, data: dict) -> dict:
        records = data.get("neuron_label_records", [])
        if not records:
            return data

        texts: list[str] = []
        labels: list[object] = []
        target_layers: list[int] = []
        models: list[str] = []
        sae_ids: list[str] = []
        tokens_nested: list[list[dict]] = []

        for rec in records:
            texts.append(str(rec.get("text", "")))
            labels.append(rec.get("label"))
            target_layers.append(int(rec["target_layer"]))
            models.append(str(rec.get("model", "")))
            sae_ids.append(str(rec.get("sae_id", "")))
            tok_list = []
            for t in rec.get("tokens", []):
                tok_list.append(
                    {
                        "token_str": str(t.get("token_str", "")),
                        "token_id": int(t["token_id"]),
                        "fired_features": [int(x) for x in t.get("fired_features", [])],
                    }
                )
            tokens_nested.append(tok_list)

        tokens_arr = pa.array(tokens_nested, type=self._TOKEN_LIST_TYPE)

        table = pa.table(
            {
                "text": texts,
                "label": pa.array(labels, type=pa.int64()),
                "target_layer": target_layers,
                "model": models,
                "sae_id": sae_ids,
                "tokens": tokens_arr,
            }
        )

        batch_path = (
            self.output_path.parent / f"{self.output_path.stem}_batch_{self.batch_counter:06d}.parquet"
        )
        pq.write_table(table, batch_path, compression="snappy")

        self.logger.info("Wrote %s rows to %s", len(records), batch_path.name)
        self.batch_counter += 1
        return data
