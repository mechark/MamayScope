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
        layer_indices = settings.resolved_mamay_important_layers()

        self.model = HookedMamay(
            important_layers=layer_indices,
            gpu_memory_utilization=0.95,
            max_model_len=2048,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            tensor_parallel_size=1, 
            download_dir="/workspace/.hf_home"
        )

    def generate_activations(self, texts: list[str]) -> List[tuple[str, list[ActivationPoint]]]:
        """
        Generate text and extract activations.

        Args:
            texts (list[str]): List of input texts to process.

        Returns:
            List[tuple[str, list[ActivationPoint]]]: List of tuples containing input text and corresponding activation points.

        Each layer ``value`` has shape ``[seq_len, hidden_size]``; last-token
        vector is ``value[-1, :]``. Longer prompts increase ``seq_len`` and memory.
        """
        all_activations = []

        for text in texts:
            _, activations_dict = self.model.generate(text)
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
    activations = service.generate_activations(texts)
    for text, acts in activations:
        print(f"Input: {text}")
        for act in acts:
            print(f"  {act.name}: shape={act.value.shape}")