import logging
from pathlib import Path
from typing import Any

from src.pipelines.base import PipelineStep


class ModelFileSink(PipelineStep):
    """Sink that saves a trained SAELens SAE to a local directory."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def run(self, data: dict) -> dict:
        if data.get("done"):
            return data

        cfg: dict[str, Any] = data.get("config") or {}
        trained_sae = data.get("trained_sae")
        if trained_sae is None:
            raise ValueError("Missing `trained_sae` in pipeline data; did the trainer step run?")

        base_dir = Path(cfg.get("output_dir", data.get("model_dir", "artifacts/sae_lens_runs/smoke")))
        subdir = str(cfg.get("inference_model_subdir", "sae_inference"))
        out_dir = base_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Saving SAE inference model to %s", out_dir)
        trained_sae.save_inference_model(str(out_dir))

        return {
            **data,
            "saved_model_dir": str(out_dir),
            "done": False,
        }

