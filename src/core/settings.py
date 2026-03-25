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

    # Neuron labeling pipeline (HF CSV + Mamay + SAELens encode → Parquet)
    NEURON_LABEL_PARQUET_PATH: str = "data/neuron_labels/labels.parquet"
    NEURON_LABEL_JSONL_PATH: str = "data/neuron_labels/labels.jsonl"
    NEURON_LABEL_DATASET_LIMIT: int = 10_000
    NEURON_LABEL_HF_DATASET: str = "mechark/controversial_statements"
    NEURON_LABEL_CSV_FILE: str = "probing_dataset.csv"
    NEURON_LABEL_TEXT_COLUMN: str = "text"
    NEURON_LABEL_LABEL_COLUMN: str = "label"
    SAE_HF_REPO_ID: str = "mechark/MamaySAE"
    SAE_HF_REVISION: str = "654d8abdaddfdcd39a34a12bbba3e332f8c11b79"
    SAE_SNAPSHOT_CACHE_DIR: str = ".cache/mamayscope/sae_snapshot"
    SAE_DEVICE: Literal["auto", "mps", "cpu", "cuda"] = "auto"
    # HF model id for AutoTokenizer (default Gemma 2 9B — override to match Mamay if seq lengths disagree)
    NEURON_LABEL_MODEL_NAME: str = "google/gemma-2-9b"
    # JSON field sae_id; if empty, derived as layer_{TARGET}_{repo_stem}_{short_rev}
    NEURON_LABEL_SAE_ID: str = ""

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