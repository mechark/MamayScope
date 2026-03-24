import logging
import os
from pathlib import Path
import pandas as pd
import torch
from src.pipelines.base import PipelineStep
from src.core.settings import settings


class ParquetSink(PipelineStep):
    """Sink that appends tensor data to a Parquet file"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_path = settings.OUTPUT_PARQUET_PATH
        
    async def run(self, data: dict) -> dict:
        """Save input/output tensors to Parquet file"""
        input_tensors = data.get("input_tensors", [])
        output_tensors = data.get("output_tensors", [])
        
        if not input_tensors or not output_tensors:
            self.logger.warning("No tensors to save")
            return data
        
        if len(input_tensors) != len(output_tensors):
            self.logger.error(
                f"Tensor count mismatch: {len(input_tensors)} inputs vs {len(output_tensors)} outputs"
            )
            return data
        
        # Ensure output directory exists
        output_dir = Path(self.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to lists for storage
        input_lists = [tensor.cpu().tolist() if isinstance(tensor, torch.Tensor) else tensor 
                       for tensor in input_tensors]
        output_lists = [tensor.cpu().tolist() if isinstance(tensor, torch.Tensor) else tensor 
                        for tensor in output_tensors]
        
        # Create DataFrame
        df = pd.DataFrame({
            "input_tensor": input_lists,
            "output_tensor": output_lists
        })
        
        # Append to parquet file
        if os.path.exists(self.output_path):
            # Read existing and append
            existing_df = pd.read_parquet(self.output_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            self.logger.info(f"Appending {len(input_lists)} rows to existing file")
        else:
            self.logger.info(f"Creating new file with {len(input_lists)} rows")
        
        df.to_parquet(self.output_path, index=False)
        
        self.logger.info(f"Saved to {self.output_path} (total rows: {len(df)})")
        
        return data
