import logging
from pathlib import Path
from typing import Any

import yaml

from src.pipelines.base import PipelineStep


class CachedActivationsConfigSource(PipelineStep):
    """Source step that loads a YAML config once and emits it downstream."""

    def __init__(self, config_path: str | Path):
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path)
        self._emitted = False

    def _load_yaml(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        raw = self.config_path.read_text(encoding="utf-8")
        cfg = yaml.safe_load(raw) or {}
        if not isinstance(cfg, dict):
            raise ValueError("Top-level YAML config must be a mapping/object")
        return cfg

    async def run(self, data: dict) -> dict:
        if self._emitted:
            return {"done": True}

        cfg = self._load_yaml()
        self._emitted = True

        return {
            "config": cfg,
            "done": False,
        }

