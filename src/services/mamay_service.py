import torch
from vllm_hook_plugins import HookedMamay
from src.schemas.activations import ActivationPoint
from src.api.schemas.schemas import ActivationRow
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

    def generate_activations(self, texts: list[str]) -> List[ActivationRow]:
        """
        Generate text and extract activations.

        Args:
            texts (list[str]): List of input texts to process.

        Returns:
            List[ActivationRow]: Per-text activations plus optional token ids used for the forward pass.

        Each layer ``value`` has shape ``[seq_len, hidden_size]``; last-token
        vector is ``value[-1, :]``. Longer prompts increase ``seq_len`` and memory.
        """
        rows: list[ActivationRow] = []

        for text in texts:
            input_ids: list[int] | None = None
            gen_with_ids = getattr(self.model, "generate_with_input_ids", None)
            if callable(gen_with_ids):
                _, activations_dict, input_ids = gen_with_ids(text)
            else:
                _, activations_dict = self.model.generate(text)
            activations = []
            
            for layer_idx, layer_acts in activations_dict.items():
                activations.append(
                    ActivationPoint(
                        name=f"layer_{layer_idx}",
                        value=layer_acts
                    )
                )

            rows.append(
                ActivationRow(
                    text=text,
                    activation_points=activations,
                    input_ids=[int(x) for x in input_ids] if input_ids is not None else None,
                )
            )

        return rows
    
if __name__ == "__main__":
    service = HookedMamayService()
    texts = ["What is the capital of France?", "Who won the World Cup in 2018?"]
    activations = service.generate_activations(texts)
    for text, acts in activations:
        print(f"Input: {text}")
        for act in acts:
            print(f"  {act.name}: shape={act.value.shape}")