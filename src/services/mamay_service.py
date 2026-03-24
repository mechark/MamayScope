import torch
from vllm_hook_plugins import HookedMamay
from src.schemas.activations import ActivationPoint
from src.core.settings import settings

from typing import List

class HookedMamayService:
    """
    Service for loading and providing access to the hooked MamayLLM running on vLLM-MamayHook.
    """
    def __init__(self):
        layer_indices = [settings.TARGET_LAYER, settings.TARGET_LAYER - 1]  # Gemma-2-9B has 42 layers

        self.model = HookedMamay(
            important_layers=layer_indices,
            gpu_memory_utilization=0.95,
            max_model_len=2048,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            tensor_parallel_size=1
        )

    def generate_activations(self, texts: list[str]) -> List[tuple[str, list[ActivationPoint]]]:
        """
        Generate text and extract activations.

        Args:
            texts (list[str]): List of input texts to process.

        Returns:
            List[tuple[str, list[ActivationPoint]]]: List of tuples containing input text and corresponding activation points.

        Activations will be of shape [hidden_size] for each layer.
        """
        all_activations = []

        for text in texts:
            _, activations_dict = self.model.generate(text) # activations is Dict[layer_idx, Tensor[hidden_size]]
            activations = []
            
            for layer_idx, layer_acts in activations_dict.items():
                activations.append(
                    ActivationPoint(
                        name=f"layer_{layer_idx}",
                        value=layer_acts
                    )
                )
                
            all_activations.append((text, activations))

        return all_activations
    
if __name__ == "__main__":
    service = HookedMamayService()
    texts = ["What is the capital of France?", "Who won the World Cup in 2018?"]
    activations = service.generate_activations_probing(texts)
    for text, acts in activations:
        print(f"Input: {text}")
        for act in acts:
            print(f"  {act.name}: shape={act.value.shape}")