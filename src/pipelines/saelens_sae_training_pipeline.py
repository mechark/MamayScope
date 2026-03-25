import argparse
import asyncio
import logging

from src.pipelines.pipeline_executor import PipelineExecutor
from src.pipelines.sources import CachedActivationsConfigSource
from src.pipelines.trainers.sae_trainer import SAELensTrainerStep
from src.pipelines.sinks import HuggingFaceHubModelSink, ModelFileSink


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def main(config_path: str) -> None:
    logger.info("Starting SAELens SAE training pipeline (cached activations)")
    logger.info("Config: %s", config_path)

    executor = PipelineExecutor(
        steps=[
            CachedActivationsConfigSource(config_path=config_path),
            SAELensTrainerStep(),
            ModelFileSink(),
            HuggingFaceHubModelSink(),
        ]
    )

    result: dict = {}
    iteration = 0
    while not result.get("done", False):
        iteration += 1
        logger.info("=== Pipeline iteration %s ===", iteration)
        result = await executor.execute({})

        if result.get("done"):
            break

    logger.info("Pipeline completed after %s iterations", iteration)
    if "saved_model_dir" in result:
        logger.info("Saved model dir: %s", result["saved_model_dir"])
    if result.get("hub_pushed"):
        logger.info("Pushed to HF repo: %s", result.get("hub_repo_id"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to trainer YAML config")
    args = parser.parse_args()

    asyncio.run(main(args.config))

