import json
import logging
from pathlib import Path

from src.core.settings import settings
from src.pipelines.base import PipelineStep


class JsonlNeuronActivationSink(PipelineStep):
    """Appends neuron labeling records as JSON lines (one object per source text)."""

    def __init__(self, output_path: str | Path | None = None):
        self.logger = logging.getLogger(__name__)
        self.output_path = Path(output_path or settings.NEURON_LABEL_JSONL_PATH)
        self.batch_counter = 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = list(self.output_path.parent.glob(f"{self.output_path.stem}_batch_*.jsonl"))
        if existing:
            self.batch_counter = max(int(f.stem.split("_batch_")[1]) for f in existing) + 1
            self.logger.info(
                "Found %s existing JSONL batch files, next batch index %s",
                len(existing),
                self.batch_counter,
            )

    async def run(self, data: dict) -> dict:
        records = data.get("neuron_label_records", [])
        if not records:
            return data

        batch_path = (
            self.output_path.parent / f"{self.output_path.stem}_batch_{self.batch_counter:06d}.jsonl"
        )
        with batch_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        self.logger.info("Wrote %s records to %s", len(records), batch_path.name)
        self.batch_counter += 1
        return data
