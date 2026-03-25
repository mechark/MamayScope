import logging
from pathlib import Path
from typing import Any

from src.pipelines.base import PipelineStep


class HuggingFaceHubModelSink(PipelineStep):
    """Sink that uploads a saved SAE directory to the Hugging Face Hub (optional)."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def run(self, data: dict) -> dict:
        if data.get("done"):
            return data

        cfg: dict[str, Any] = data.get("config") or {}
        push_to_hub = bool(cfg.get("push_to_hub", False))

        saved_model_dir = data.get("saved_model_dir")
        if saved_model_dir is None:
            raise ValueError("Missing `saved_model_dir` in pipeline data; did ModelFileSink run?")

        if not push_to_hub:
            self.logger.info("push_to_hub=false; skipping Hub upload.")
            return {**data, "hub_pushed": False, "done": True}

        repo_id = cfg.get("hf_repo_id")
        if not repo_id:
            raise ValueError("push_to_hub=true but `hf_repo_id` is missing from config")

        private = bool(cfg.get("hf_private", False))
        revision = cfg.get("hf_revision")  # optional branch/tag
        commit_message = str(cfg.get("hf_commit_message", "Upload trained SAE"))

        from huggingface_hub import HfApi

        api = HfApi()
        self.logger.info("Creating/updating HF model repo: %s (private=%s)", repo_id, private)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

        folder_path = Path(saved_model_dir)
        if not folder_path.exists():
            raise FileNotFoundError(f"Saved model directory not found: {folder_path}")

        self.logger.info("Uploading folder to Hub: %s -> %s", folder_path, repo_id)
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=str(repo_id),
            repo_type="model",
            revision=revision,
            commit_message=commit_message,
        )

        return {**data, "hub_pushed": True, "hub_repo_id": str(repo_id), "done": True}

