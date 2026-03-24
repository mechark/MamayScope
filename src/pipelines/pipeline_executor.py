import logging
from src.pipelines.base import PipelineStep

logging.basicConfig(level=logging.INFO)

class PipelineExecutor:
    """Executor that takes a list of pipeline steps and runs them sequentially, passing the output of one step as the input to the next"""
    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps
        self.logger = logging.getLogger(__name__)

    async def execute(self, initial_data: dict) -> dict:
        data = initial_data
        for step in self.steps:
            self.logger.info(f"Running step: {step.__class__.__name__}")
            data = await step.run(data)

        self.logger.info("Pipeline execution completed.")
        return data