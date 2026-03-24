from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings"""
    MODEL_ENDPOINT: str
    TARGET_LAYER: int = 33
    BATCH_SIZE: int = 32
    DATASET_LIMIT: int = 100000
    OUTPUT_PARQUET_PATH: str = "data/sae_training.parquet"
   
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_nested_delimiter="__"
    )

settings = Settings()