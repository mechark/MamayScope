from pydantic import BaseModel, Field
from src.schemas.activations import ActivationPoint


class ActivationRequest(BaseModel):
    """Request schema for activations endpoint"""
    texts: list[str] = Field(..., description="List of input texts to process", min_length=1)


class ActivationResponse(BaseModel):
    """Response schema for activations endpoint"""
    activations: list[tuple[str, list[ActivationPoint]]] = Field(..., description="List of activation points with their values")
