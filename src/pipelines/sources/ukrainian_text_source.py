import logging
from typing import Iterator
from datasets import load_dataset
from src.pipelines.base import PipelineStep
from src.core.settings import settings


class UkrainianTextSource(PipelineStep):
    """Source that yields batches of Ukrainian user texts from Hermes3-UK dataset"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dataset_iterator: Iterator | None = None
        self.processed_count = 0
        self.limit = settings.DATASET_LIMIT
        self.batch_size = settings.BATCH_SIZE
        
    def _initialize_iterator(self):
        """Initialize the dataset iterator"""
        if self.dataset_iterator is None:
            self.logger.info("Loading lapa-llm/hermes3-uk dataset in streaming mode...")
            ds = load_dataset("lapa-llm/hermes3-uk", split="train", streaming=True)
            
            # Filter for user messages and extract text values
            def extract_user_texts(example):
                """Extract user messages from conversations"""
                conversations = example.get("conversations", [])
                user_texts = []
                for message in conversations:
                    if isinstance(message, dict) and message.get("from") == "user":
                        value = message.get("value")
                        if value:
                            user_texts.append(value)
                return {"user_texts": user_texts}
            
            # Map to extract user texts and flatten
            ds = ds.map(extract_user_texts)
            
            # Create iterator that yields individual texts
            def text_generator():
                for item in ds:
                    user_texts = item.get("user_texts", [])
                    for text in user_texts:
                        yield text
            
            self.dataset_iterator = text_generator()
            self.logger.info("Dataset iterator initialized")
    
    async def run(self, data: dict) -> dict:
        """Yield a batch of texts from the dataset"""
        self._initialize_iterator()
        
        texts = []
        done = False
        
        # Collect batch_size texts or until limit reached
        for _ in range(self.batch_size):
            if self.processed_count >= self.limit:
                done = True
                break
            
            try:
                text = next(self.dataset_iterator)
                texts.append(text)
                self.processed_count += 1
            except StopIteration:
                done = True
                break
        
        self.logger.info(
            f"Extracted {len(texts)} texts (total: {self.processed_count}/{self.limit})"
        )
        
        return {
            "texts": texts,
            "done": done or len(texts) == 0
        }
