from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.core.settings import settings
from src.services.model_service import ModelService
from src.api.schemas.schemas import ActivationRequest, ActivationResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model service instance
model_service: ModelService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for loading model on startup"""
    global model_service
    
    logger.info(f"Loading model: {settings.MODEL_NAME}")
    try:
        # Initialize with empty activation points - they'll be provided per request
        model_service = ModelService(model_name=settings.MODEL_NAME, activation_points=[])
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down application")
    model_service = None


# Create FastAPI application
app = FastAPI(
    title="MamayScope API",
    description="API for extracting activations from transformer models",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware to accept requests from everyone
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": settings.MODEL_NAME,
        "message": "MamayScope API is running"
    }


@app.post("/activations", response_model=ActivationResponse)
async def get_activations(request: ActivationRequest):
    """
    Extract activations from the model for given texts and activation points.
    
    Args:
        request: ActivationRequest containing texts and activation_points
    
    Returns:
        ActivationResponse containing the extracted activations
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model service is not initialized")
    
    try:
        logger.info(f"Processing {len(request.texts)} texts with {len(request.activation_points)} activation points")
        
        # Update activation points for this request
        model_service.activation_points = request.activation_points
        
        # Get activations
        activations = model_service.get_activations(request.texts)
        
        return ActivationResponse(activations=activations)
    
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid activation point: {str(e)}. Please check that the activation point exists in the model."
        )
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
