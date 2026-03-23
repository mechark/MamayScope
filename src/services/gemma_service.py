import torch
from transformers import AutoModelForCausalLM
from src.schemas.activations import ActivationPoint

class HookedGemma:
    """
    Service for loading and providing access to LLM hooked models.
    """
    def __init__(self, model_name: str, activation_points: list[str]):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.activations = dict()

    def get_activations(self, input_text: list[str]) -> list[ActivationPoint]:
        """
        Run the model with caching to get activations.

        Args:
            input_text (list[str]): List of input texts to process.

        Returns:
            list[ActivationPoint]: List of activation points with their values.

        Activations will be of shape [batch_size, sequence_length, model_dimension]
        """
        logits, cache = self.model.run_with_cache(input_text)
        activations = [ActivationPoint(name=point, value=cache[point]) for point in self.activation_points]
        return activations