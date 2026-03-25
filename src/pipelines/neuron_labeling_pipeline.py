import asyncio
import logging

from src.core.settings import settings
from src.pipelines.pipeline_executor import PipelineExecutor
from src.pipelines.processors import MamayActivationProcessor, SaeFeatureEncodeProcessor
from src.pipelines.sinks import ParquetNeuronActivationSink
from src.pipelines.sources import HFCsvBatchSource

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("Neuron labeling pipeline (HF CSV → Mamay → SAELens → Parquet)")
    logger.info(
        "TARGET_LAYER=%s BATCH_SIZE=%s NEURON_LABEL_DATASET_LIMIT=%s MODEL_ENDPOINT=%s SAE_DEVICE=%s",
        settings.TARGET_LAYER,
        settings.BATCH_SIZE,
        settings.NEURON_LABEL_DATASET_LIMIT,
        settings.MODEL_ENDPOINT,
        settings.SAE_DEVICE,
    )
    logger.info(
        "NEURON_LABEL_MODEL_NAME=%r NEURON_LABEL_SAE_ID=%r NEURON_LABEL_PARQUET_PATH=%s",
        settings.NEURON_LABEL_MODEL_NAME,
        settings.NEURON_LABEL_SAE_ID,
        settings.NEURON_LABEL_PARQUET_PATH,
    )

    executor = PipelineExecutor(
        steps=[
            HFCsvBatchSource(),
            MamayActivationProcessor(),
            SaeFeatureEncodeProcessor(),
            ParquetNeuronActivationSink(),
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
