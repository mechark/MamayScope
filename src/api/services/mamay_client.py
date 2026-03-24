import httpx
from typing import List
import torch
from src.schemas.activations import ActivationPoint
from src.core.settings import settings

class MamayClient:
    """
    Async client for making requests to the MamayScope API endpoint.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the Mamay client.
        
        Args:
            base_url (str): Base URL of the MamayScope API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes the client."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def health_check(self) -> dict:
        """
        Check if the API is healthy and running.
        
        Returns:
            dict: Health check response with status and service information
        
        Raises:
            httpx.HTTPError: If the request fails
        """
        response = await self.client.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()
    
    async def get_activations(
        self, 
        texts: List[str]
    ) -> List[tuple[str, List[ActivationPoint]]]:
        """
        Get activations from the MamayScope API.
        
        Args:
            texts (List[str]): List of input texts to process
        
        Returns:
            List[tuple[str, List[ActivationPoint]]]: List of tuples containing 
                input text and corresponding activation points
        
        Raises:
            httpx.HTTPError: If the request fails
        """
        request_data = {
            "texts": texts
        }
        
        response = await self.client.post(
            f"{self.base_url}/activations",
            json=request_data
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Parse the response into ActivationPoint objects
        activations = []
        for text, acts_data in data["activations"]:
            activation_points = []
            for act in acts_data:
                # Reconstruct tensor from serialized format
                tensor_data = act["value"]
                tensor = torch.tensor(tensor_data["data"])
                
                activation_points.append(
                    ActivationPoint(
                        name=act["name"],
                        value=tensor
                    )
                )
            activations.append((text, activation_points))
        
        return activations


if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example usage of async client
        async with MamayClient(settings.MODEL_ENDPOINT) as client:
            # Health check
            health = await client.health_check()
            print(f"Health check: {health}")
            
            # Get activations
            texts = ["What is the capital of France?", "Who won the World Cup in 2018?"]
            activations = await client.get_activations(texts)
            
            for text, acts in activations:
                print(f"\nInput: {text}")
                print(f"Number of layers: {len(acts)}")
                for act in acts[:3]:  # Show first 3 layers
                    print(f"  {act.name}: shape={act.value.shape}")
    
    asyncio.run(main())