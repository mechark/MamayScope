import asyncio
import logging
from pathlib import Path

from src.core.settings import settings
from src.pipelines.pipeline_executor import PipelineExecutor
from src.pipelines.political_corpus_pipeline import ParquetTextBatchSource
from src.pipelines.processors import MamayActivationProcessor
from src.pipelines.sinks import ParquetSink
from src.pipelines.sources import UkrainianTextSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Main pipeline execution"""
    logger.info("Starting SAE training data generation pipeline...")
    logger.info(
        "Configuration: TARGET_LAYER=%s, BATCH_SIZE=%s, DATASET_LIMIT=%s, SAE_TEXT_SOURCE=%s",
        settings.TARGET_LAYER,
        settings.BATCH_SIZE,
        settings.DATASET_LIMIT,
        settings.SAE_TEXT_SOURCE,
    )

    if settings.SAE_TEXT_SOURCE == "parquet":
        corpus = Path(settings.SAE_CORPUS_PARQUET_PATH)
        logger.info(
            "Text source: Parquet corpus at %s (column=%s, skip_rows=%s)",
            corpus,
            settings.SAE_CORPUS_TEXT_COLUMN,
            settings.SAE_CORPUS_SKIP_ROWS,
        )
        first_step = ParquetTextBatchSource(
            parquet_path=corpus,
            text_column=settings.SAE_CORPUS_TEXT_COLUMN,
            batch_size=settings.BATCH_SIZE,
            max_rows=settings.DATASET_LIMIT,
            skip_rows=settings.SAE_CORPUS_SKIP_ROWS,
            language_column=None,
            language_filter=None,
        )
    else:
        logger.info(
            "Text source: Hermes3-UK streaming (skip_count=%s)",
            settings.SAE_HERMES_SKIP_COUNT,
        )
        first_step = UkrainianTextSource(skip_count=settings.SAE_HERMES_SKIP_COUNT)

    executor = PipelineExecutor(
        steps=[
            first_step,
            MamayActivationProcessor(),
            ParquetSink(),
        ]
    )
    
    # Run pipeline in loop until done
    result = {}
    iteration = 0
    
    while not result.get("done", False):
        iteration += 1
        logger.info(f"=== Pipeline iteration {iteration} ===")
        result = await executor.execute({})
        
        if result.get("done"):
            logger.info("Pipeline completed!")
            break
    
    logger.info(f"Finished after {iteration} iterations")


if __name__ == "__main__":
    asyncio.run(main())
