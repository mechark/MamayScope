import torch
from src.schemas.activations import ActivationPoint
from typing import List


class MockMamayService:
    """
    Mock service for MamayLLM that generates random tensors for testing.
    Mimics the behavior of HookedMamayService without requiring actual model loading.
    """
    
    def __init__(self, num_layers: int = 42, hidden_size: int = 3584):
        """
        Initialize the mock service.
        
        Args:
            num_layers (int): Number of layers to simulate (default: 42 for Gemma-2-9B)
            hidden_size (int): Hidden dimension size (default: 3584 for Gemma-2-9B)
        """
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
    def generate_activations(self, texts: list[str]) -> List[tuple[str, list[ActivationPoint]]]:
        """
        Generate mock activations with random tensors.
        
        Args:
            texts (list[str]): List of input texts to process.
            
        Returns:
            List[tuple[str, list[ActivationPoint]]]: List of tuples containing input text 
            and corresponding random activation points.
        """
        all_activations = []
        
        for text in texts:
            activations = []
            
            # Generate random activations for each layer
            for layer_idx in range(self.num_layers):
                # Generate random tensor with shape [hidden_size]
                random_tensor = torch.randn(self.hidden_size)
                
                activations.append(
                    ActivationPoint(
                        name=f"layer_{layer_idx}",
                        value=random_tensor
                    )
                )
            
            all_activations.append((text, activations))
        
        return all_activations


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