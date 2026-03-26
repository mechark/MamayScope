import asyncio
import logging

from src.core.settings import settings
from src.pipelines.pipeline_executor import PipelineExecutor
from src.pipelines.processors import MamayActivationProcessor, SaeFeatureEncodeProcessor
from src.pipelines.sinks import ParquetNeuronActivationSink
from src.pipelines.sources import ParquetConversationBatchSource

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("Neuron labeling pipeline (Parquet conversations → Mamay → SAELens → Parquet)")
    logger.info(
        "TARGET_LAYER=%s BATCH_SIZE=%s MODEL_ENDPOINT=%s SAE_DEVICE=%s",
        settings.TARGET_LAYER,
        settings.BATCH_SIZE,
        settings.MODEL_ENDPOINT,
        settings.SAE_DEVICE,
    )
    logger.info(
        "NEURON_LABEL_MODEL_NAME=%r NEURON_LABEL_SAE_ID=%r OUTPUT_PATH=%s",
        settings.NEURON_LABEL_MODEL_NAME,
        settings.NEURON_LABEL_SAE_ID,
        settings.NEURON_LABEL_PROPAGANDA_OUTPUT_PARQUET_PATH,
    )
    logger.info(
        "PROPAGANDA_SOURCE=%s CONVERSATION_COLUMN=%s SOURCE_BATCH_SIZE=%s SOURCE_MAX_ROWS=%s DEDUP_EXACT=%s",
        settings.NEURON_LABEL_PROPAGANDA_PARQUET_SOURCE_PATH,
        settings.NEURON_LABEL_PROPAGANDA_CONVERSATION_COLUMN,
        settings.NEURON_LABEL_PROPAGANDA_BATCH_SIZE,
        settings.NEURON_LABEL_PROPAGANDA_MAX_ROWS,
        settings.NEURON_LABEL_PROPAGANDA_DEDUP_EXACT,
    )

    executor = PipelineExecutor(
        steps=[
            ParquetConversationBatchSource(),
            MamayActivationProcessor(),
            SaeFeatureEncodeProcessor(),
            ParquetNeuronActivationSink(
                output_path=settings.NEURON_LABEL_PROPAGANDA_OUTPUT_PARQUET_PATH
            ),
        ]
    )

    result: dict = {}
    iteration = 0
    while not result.get("done", False):
        iteration += 1
        logger.info("=== Pipeline iteration %s ===", iteration)
        result = await executor.execute({})
        if result.get("done"):
            logger.info("Pipeline completed.")
            break

    logger.info("Finished after %s iterations", iteration)


if __name__ == "__main__":
    asyncio.run(main())
