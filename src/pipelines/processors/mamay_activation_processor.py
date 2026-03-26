import logging
import asyncio
import torch
import httpx
from src.pipelines.base import PipelineStep
from src.api.services.mamay_client import MamayClient
from src.core.settings import settings


class MamayActivationProcessor(PipelineStep):
    """Processor that fetches activations from Mamay API and extracts layer input/output tensors"""
    
    def __init__(self, parallel_requests: int | None = None):
        self.logger = logging.getLogger(__name__)
        self.client = MamayClient(base_url=settings.MODEL_ENDPOINT)
        self.target_layer = settings.TARGET_LAYER
        self.parallel_requests = parallel_requests or settings.PARALLEL_REQUESTS
        self.max_retries = settings.MAX_RETRIES
        self.retry_delay = settings.RETRY_DELAY
    
    async def _get_activations_with_retry(self, chunk: list[str], chunk_idx: int) -> list:
        """Get activations with retry; on persistent 5xx split chunk to isolate bad rows."""
        if not chunk:
            return []

        async def _split_now() -> list:
            if len(chunk) == 1:
                return []
            mid = len(chunk) // 2
            self.logger.warning(
                "Chunk %s received 5xx; splitting %s -> %s + %s to isolate bad rows",
                chunk_idx,
                len(chunk),
                mid,
                len(chunk) - mid,
            )
            left = await self._get_activations_with_retry(chunk[:mid], chunk_idx * 2)
            right = await self._get_activations_with_retry(chunk[mid:], chunk_idx * 2 + 1)
            return left + right

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await self.client.get_activations(chunk)
            except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                last_exc = e
            except httpx.HTTPStatusError as e:
                last_exc = e
                # For server-side row failures (HTTP 5xx), split large chunks immediately.
                status = e.response.status_code if e.response is not None else None
                if status is not None and status < 500:
                    raise
                if status is not None and status >= 500 and len(chunk) > 1:
                    return await _split_now()

            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                self.logger.warning(
                    f"Chunk {chunk_idx} failed (attempt {attempt + 1}/{self.max_retries}): {last_exc}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)

        # Persistent failure: isolate problematic text(s) by splitting.
        if len(chunk) > 1:
            return await _split_now()

        # Single-row failure: preserve pipeline progress with empty activations for this text.
        text = chunk[0] if chunk else ""
        self.logger.error(
            "Single text failed after retries in chunk %s; returning empty activations for text=%r",
            chunk_idx,
            text[:120],
        )
        return [(text, [], None)]
        
    async def run(self, data: dict) -> dict:
        """Fetch activations and extract input/output tensors for target layer"""
        texts = data.get("texts", [])
        labels_in = data.get("labels")
        if labels_in is not None and len(labels_in) != len(texts):
            self.logger.warning(
                "labels length %s != texts length %s; dropping labels",
                len(labels_in),
                len(texts),
            )
            labels_in = None

        if not texts:
            self.logger.warning("No texts to process")
            out_empty: dict = {
                "input_tensors": [],
                "output_tensors": [],
                "done": data.get("done", False),
                "texts": [],
            }
            if labels_in is not None:
                out_empty["labels"] = []
            return out_empty
        
        # Split texts into chunks for parallel processing
        num_parallel = min(self.parallel_requests, len(texts))
        chunk_size = (len(texts) + num_parallel - 1) // num_parallel
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        self.logger.info(
            f"Fetching activations for {len(texts)} texts in {len(chunks)} parallel requests "
            f"(~{chunk_size} texts each)..."
        )
        
        # Get activations from API in parallel with retry logic
        chunk_results = await asyncio.gather(
            *[self._get_activations_with_retry(chunk, i) for i, chunk in enumerate(chunks)]
        )
        
        # Flatten results
        activations_data = []
        for result in chunk_results:
            activations_data.extend(result)
        
        input_tensors: list[torch.Tensor] = []
        output_tensors: list[torch.Tensor] = []
        source_texts: list[str] = []
        token_indices: list[int] = []
        labels_per_row: list = []
        input_ids_per_row: list[list[int] | None] = []
        h = settings.HIDDEN_SIZE
        flatten = settings.PIPELINE_FLATTEN_TOKENS

        input_layer_name = f"layer_{self.target_layer - 1}"
        output_layer_name = f"layer_{self.target_layer}"

        def _empty_2d() -> torch.Tensor:
            return torch.zeros(0, h, dtype=torch.float32)

        def _as_2d(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 1:
                return t.unsqueeze(0)
            return t

        for row_idx, (text, activation_points, input_ids) in enumerate(activations_data):
            activation_dict = {ap.name: ap.value for ap in activation_points}
            row_label = labels_in[row_idx] if labels_in is not None else None

            if input_layer_name in activation_dict:
                input_tensor = activation_dict[input_layer_name].float()
            else:
                self.logger.warning(
                    f"Missing {input_layer_name} for text: {text[:50]}..."
                )
                input_tensor = _empty_2d()

            if output_layer_name in activation_dict:
                output_tensor = activation_dict[output_layer_name].float()
            else:
                self.logger.warning(
                    f"Missing {output_layer_name} for text: {text[:50]}..."
                )
                output_tensor = _empty_2d()

            input_tensor = _as_2d(input_tensor)
            output_tensor = _as_2d(output_tensor)

            if input_tensor.shape[0] != output_tensor.shape[0]:
                self.logger.warning(
                    "Input/output sequence length mismatch for text %r: "
                    "layer %s dim0=%s vs %s dim0=%s",
                    text[:80],
                    input_layer_name,
                    input_tensor.shape[0],
                    output_layer_name,
                    output_tensor.shape[0],
                )

            if flatten:
                n = min(input_tensor.shape[0], output_tensor.shape[0])
                for t_idx in range(n):
                    input_tensors.append(input_tensor[t_idx].contiguous())
                    output_tensors.append(output_tensor[t_idx].contiguous())
                    source_texts.append(text)
                    token_indices.append(t_idx)
                    labels_per_row.append(row_label)
                    # Per-token flattening: store full sequence ids for each token row (or None).
                    input_ids_per_row.append(list(input_ids) if input_ids is not None else None)
            else:
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
                input_ids_per_row.append(list(input_ids) if input_ids is not None else None)

        self.logger.info(
            f"Extracted {len(input_tensors)} input/output tensor pairs for layer {self.target_layer}"
            + (" (flattened per token)" if flatten else "")
        )

        out: dict = {
            "input_tensors": input_tensors,
            "output_tensors": output_tensors,
            "done": data.get("done", False),
        }
        if flatten and source_texts:
            out["source_texts"] = source_texts
            out["token_indices"] = token_indices
            out["texts"] = source_texts
            out["labels"] = labels_per_row
        else:
            out["texts"] = list(texts)
            if labels_in is not None:
                out["labels"] = list(labels_in)

        out["input_ids"] = input_ids_per_row
        return out
