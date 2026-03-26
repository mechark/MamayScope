from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.services.mock_mamay_service import MockMamayService
from src.api.schemas.schemas import ActivationRequest, ActivationResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model service instance
model_service: MockMamayService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for loading model on startup"""
    global model_service
    
    use_mock = os.environ.get("MAMAY_USE_MOCK", "").lower() in (
        "1",
        "true",
        "yes",
    )
    try:
        if use_mock:
            logger.info("Initializing MockMamayService (MAMAY_USE_MOCK set)")
            model_service = MockMamayService()
        else:
            from src.services.mamay_service import HookedMamayService

            logger.info("Initializing HookedMamayService")
            model_service = HookedMamayService()
        logger.info("Model service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
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
    name = type(model_service).__name__ if model_service else "none"
    return {
        "status": "healthy",
        "service": name,
        "message": "MamayScope API is running",
    }


@app.post("/activations", response_model=ActivationResponse)
async def get_activations(request: ActivationRequest):
    """
    Extract activations from the mock model for given texts.
    
    Args:
        request: ActivationRequest containing texts to process.
    
    Returns:
        ActivationResponse containing the mock activations (random tensors)
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model service is not initialized")
    
    try:
        logger.info(f"Processing {len(request.texts)} texts (mock service generates all layers)")
        
        # Generate mock activations for all texts
        # Model service returns List[ActivationRow]
        activations = model_service.generate_activations(request.texts)
        
        return ActivationResponse(activations=activations)
    
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
