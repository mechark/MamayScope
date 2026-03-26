import httpx
from typing import List, Optional, Union, Any
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
        self.client = httpx.AsyncClient(timeout=300.0)
    
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
    ) -> List[tuple[str, List[ActivationPoint], Optional[list[int]]]]:
        """
        Get activations from the MamayScope API.
        
        Args:
            texts (List[str]): List of input texts to process
        
        Returns:
            List[tuple[str, List[ActivationPoint], Optional[list[int]]]]: Per input text, activation points,
            and optional full-sequence token ids aligned to activations time dim.
        
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
        
        def _parse_points(acts_data: Any) -> list[ActivationPoint]:
            activation_points: list[ActivationPoint] = []
            for act in acts_data:
                tensor_data = act["value"]
                if isinstance(tensor_data, dict):
                    raw = tensor_data["data"]
                    shape = tensor_data.get("shape")
                    tensor = torch.as_tensor(raw, dtype=torch.float32)
                    if shape:
                        tensor = tensor.reshape(*tuple(shape))
                else:
                    tensor = torch.as_tensor(tensor_data, dtype=torch.float32)

                activation_points.append(
                    ActivationPoint(
                        name=act["name"],
                        value=tensor,
                    )
                )
            return activation_points

        out: list[tuple[str, list[ActivationPoint], Optional[list[int]]]] = []
        rows = data.get("activations", [])

        # Backward compatibility:
        # - Old server: {"activations": [[text, acts_data], ...]}
        # - New server: {"activations": [{"text":..., "activation_points":..., "input_ids":...}, ...]}
        for row in rows:
            if isinstance(row, dict):
                text = str(row.get("text", ""))
                acts_data = row.get("activation_points", [])
                input_ids = row.get("input_ids")
                parsed_ids = [int(x) for x in input_ids] if isinstance(input_ids, list) else None
                out.append((text, _parse_points(acts_data), parsed_ids))
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                text = str(row[0])
                acts_data = row[1]
                input_ids = row[2] if len(row) >= 3 else None
                parsed_ids = [int(x) for x in input_ids] if isinstance(input_ids, list) else None
                out.append((text, _parse_points(acts_data), parsed_ids))
            else:
                raise TypeError(f"Unexpected activations row type: {type(row)}")

        return out


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