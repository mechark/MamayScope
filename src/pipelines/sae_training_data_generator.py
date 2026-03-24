import asyncio
import logging
from src.pipelines.pipeline_executor import PipelineExecutor
from src.pipelines.sources import UkrainianTextSource
from src.pipelines.processors import MamayActivationProcessor
from src.pipelines.sinks import ParquetSink
from src.core.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Main pipeline execution"""
    logger.info("Starting SAE training data generation pipeline...")
    logger.info(f"Configuration: TARGET_LAYER={settings.TARGET_LAYER}, "
                f"BATCH_SIZE={settings.BATCH_SIZE}, "
                f"DATASET_LIMIT={settings.DATASET_LIMIT}")
    
    # Create executor
    executor = PipelineExecutor(
        UkrainianTextSource(),
        MamayActivationProcessor(),
        ParquetSink()
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
