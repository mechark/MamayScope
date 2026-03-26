from pydantic import BaseModel, Field
from src.schemas.activations import ActivationPoint


class ActivationRequest(BaseModel):
    """Request schema for activations endpoint"""
    texts: list[str] = Field(..., description="List of input texts to process", min_length=1)

class ActivationRow(BaseModel):
    """Per-text activations + optional token IDs used for the forward pass."""

    text: str = Field(..., description="Original input text")
    activation_points: list[ActivationPoint] = Field(..., description="Activation points for this text")
    input_ids: list[int] | None = Field(
        default=None,
        description="Optional full-sequence token ids (prompt + generated continuation), aligned to activations time dim.",
    )


class ActivationResponse(BaseModel):
    """Response schema for activations endpoint"""
    activations: list[ActivationRow] = Field(
        ...,
        description=(
            "Per-text activation rows; each ``value`` may be [H] or [seq_len, H] "
            "(serialized with shape + flat data)."
        ),
    )
