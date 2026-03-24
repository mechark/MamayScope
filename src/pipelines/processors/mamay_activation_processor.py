import logging
import torch
from src.pipelines.base import PipelineStep
from src.api.services.mamay_client import MamayClient
from src.core.settings import settings


class MamayActivationProcessor(PipelineStep):
    """Processor that fetches activations from Mamay API and extracts layer input/output tensors"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = MamayClient(base_url=settings.MODEL_ENDPOINT)
        self.target_layer = settings.TARGET_LAYER
        
    async def run(self, data: dict) -> dict:
        """Fetch activations and extract input/output tensors for target layer"""
        texts = data.get("texts", [])
        
        if not texts:
            self.logger.warning("No texts to process")
            return {
                "input_tensors": [],
                "output_tensors": [],
                "done": data.get("done", True)
            }
        
        self.logger.info(f"Fetching activations for {len(texts)} texts...")
        
        # Get activations from API
        activations_data = await self.client.get_activations(texts)
        
        input_tensors = []
        output_tensors = []
        
        # Extract input (layer N-1) and output (layer N) tensors
        input_layer_name = f"layer_{self.target_layer - 1}"
        output_layer_name = f"layer_{self.target_layer}"
        
        for text, activation_points in activations_data:
            # Create lookup dict for faster access
            activation_dict = {ap.name: ap.value for ap in activation_points}
            
            # Get input tensor (output of previous layer)
            if input_layer_name in activation_dict:
                input_tensor = activation_dict[input_layer_name]
                input_tensors.append(input_tensor)
            else:
                self.logger.warning(f"Missing {input_layer_name} for text: {text[:50]}...")
                input_tensors.append(torch.zeros(3584))  # Fallback
            
            # Get output tensor (output of target layer)
            if output_layer_name in activation_dict:
                output_tensor = activation_dict[output_layer_name]
                output_tensors.append(output_tensor)
            else:
                self.logger.warning(f"Missing {output_layer_name} for text: {text[:50]}...")
                output_tensors.append(torch.zeros(3584))  # Fallback
        
        self.logger.info(
            f"Extracted {len(input_tensors)} input/output tensor pairs for layer {self.target_layer}"
        )
        
        return {
            "input_tensors": input_tensors,
            "output_tensors": output_tensors,
            "done": data.get("done", False)
        }
