from abc import ABC, abstractmethod

class PipelineStep(ABC):
    @abstractmethod
    async def run(self, data: dict) -> dict:
        pass