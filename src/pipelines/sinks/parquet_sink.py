import logging
import os
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from src.pipelines.base import PipelineStep
from src.core.settings import settings


class ParquetSink(PipelineStep):
    """Sink that appends tensor data to a Parquet file"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_path = Path(settings.OUTPUT_PARQUET_PATH)
        self.batch_counter = 0
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Count existing batch files to continue numbering
        existing_files = list(self.output_path.parent.glob(f"{self.output_path.stem}_batch_*.parquet"))
        if existing_files:
            self.batch_counter = max([
                int(f.stem.split('_batch_')[1]) 
                for f in existing_files
            ]) + 1
            self.logger.info(f"Found {len(existing_files)} existing batch files, starting from batch {self.batch_counter}")
        
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
        input_lists = [
            tensor.cpu().tolist() if isinstance(tensor, torch.Tensor) else tensor
            for tensor in input_tensors
        ]
        output_lists = [
            tensor.cpu().tolist() if isinstance(tensor, torch.Tensor) else tensor
            for tensor in output_tensors
        ]

        cols: dict = {
            "input_tensor": input_lists,
            "output_tensor": output_lists,
        }
        st = data.get("source_texts")
        ti = data.get("token_indices")
        if st is not None and ti is not None:
            if len(st) != len(input_lists) or len(ti) != len(input_lists):
                self.logger.error(
                    "Provenance column length mismatch vs tensors; skipping those columns"
                )
            else:
                cols["source_text"] = st
                cols["token_idx"] = ti

        df = pd.DataFrame(cols)
        
        # Convert to PyArrow table
        table = pa.Table.from_pandas(df)
        
        # Write to a new batch file (no memory leak, preserves all data)
        batch_path = self.output_path.parent / f"{self.output_path.stem}_batch_{self.batch_counter:06d}.parquet"
        pq.write_table(table, batch_path, compression='snappy')
        
        self.logger.info(f"Saved {len(input_lists)} rows to {batch_path.name}")
        self.batch_counter += 1
        
        # Clean up memory
        del df, table, input_lists, output_lists
        
        return data
