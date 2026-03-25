from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings"""
    MODEL_ENDPOINT: str
    TARGET_LAYER: int = 33
    BATCH_SIZE: int = 32
    DATASET_LIMIT: int = 100000
    OUTPUT_PARQUET_PATH: str = "data/politics_activations/sae_training_fullseq.parquet"
    SAE_TEXT_SOURCE: Literal["hermes", "parquet"] = "hermes"
    SAE_CORPUS_PARQUET_PATH: str = "data/corpus"
    SAE_CORPUS_TEXT_COLUMN: str = "text"
    SAE_CORPUS_SKIP_ROWS: int = 0
    SAE_HERMES_SKIP_COUNT: int = 100_000 + 792 + 224 + 544
    PARALLEL_REQUESTS: int = 4
    HIDDEN_SIZE: int = 3584
    MAMAY_IMPORTANT_LAYERS: str | None = None
    PIPELINE_FLATTEN_TOKENS: bool = False
    MOCK_SEQ_LEN: int = 8
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    def resolved_mamay_important_layers(self) -> list[int]:
        """Layer indices for HookedMamay: env comma-list or [TARGET_LAYER-1, TARGET_LAYER]."""
        raw = self.MAMAY_IMPORTANT_LAYERS
        if raw and raw.strip():
            return [
                int(x.strip())
                for x in raw.split(",")
                if x.strip()
            ]
        t = self.TARGET_LAYER
        return [t - 1, t]


settings = Settings()