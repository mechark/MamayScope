import torch
from src.schemas.activations import ActivationPoint
from src.api.schemas.schemas import ActivationRow
from src.core.settings import settings
from typing import List


class MockMamayService:
    """
    Mock service for MamayLLM that generates random tensors for testing.
    Mimics the behavior of HookedMamayService without requiring actual model loading.
    """
    
    def __init__(
        self,
        num_layers: int = 42,
        hidden_size: int | None = None,
        seq_len: int | None = None,
    ):
        """
        Initialize the mock service.

        Args:
            num_layers: Number of layers to simulate (default: 42 for Gemma-2-9B).
            hidden_size: Hidden dim; defaults to ``settings.HIDDEN_SIZE``.
            seq_len: Sequence length for 2D activations; defaults to ``settings.MOCK_SEQ_LEN``.
        """
        self.num_layers = num_layers
        self.hidden_size = hidden_size if hidden_size is not None else settings.HIDDEN_SIZE
        self.seq_len = seq_len if seq_len is not None else settings.MOCK_SEQ_LEN
        
    def generate_activations(self, texts: list[str]) -> List[ActivationRow]:
        """
        Generate mock activations with random tensors.
        
        Args:
            texts (list[str]): List of input texts to process.
            
        Returns:
            List[ActivationRow]: Per-text activation rows (no real token ids in mock mode).
        """
        rows: list[ActivationRow] = []
        
        for text in texts:
            activations = []
            
            # Generate random activations for each layer
            for layer_idx in range(self.num_layers):
                random_tensor = torch.randn(self.seq_len, self.hidden_size)
                
                activations.append(
                    ActivationPoint(
                        name=f"layer_{layer_idx}",
                        value=random_tensor
                    )
                )
            
            rows.append(
                ActivationRow(
                    text=text,
                    activation_points=activations,
                    input_ids=None,
                )
            )
        
        return rows


if __name__ == "__main__":
    # Test the mock service
    service = MockMamayService()
    texts = ["What is the capital of France?", "Who won the World Cup in 2018?"]
    activations = service.generate_activations(texts)
    
    for text, acts in activations:
        print(f"Input: {text}")
        for act in acts[:3]:  # Print first 3 layers only
            print(f"  {act.name}: shape={act.value.shape}, mean={act.value.mean():.4f}, std={act.value.std():.4f}")
        print(f"  ... ({len(acts)} layers total)")