from pydantic import BaseModel, Field, field_serializer
import torch


class ActivationPoint(BaseModel):
    """Response model for activation endpoint"""
    name: str = Field(..., description="Name of the activation point")
    value: torch.Tensor = Field(..., description="Activation values as a tensor")
    
    @field_serializer('value')
    def serialize_tensor(self, tensor: torch.Tensor) -> dict:
        """Serialize torch.Tensor to JSON-compatible format"""
        return {
            "shape": list(tensor.shape),
            "data": tensor.tolist()
        }
    
    model_config = {"arbitrary_types_allowed": True}