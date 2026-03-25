import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from src.pipelines.base import PipelineStep


class SAELensTrainerStep(PipelineStep):
    """Trainer step that trains a SAELens SAE on cached activations saved with datasets.save_to_disk()."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _activation_iterator_from_cached_dataset(
        self,
        cfg: dict[str, Any],
        hook_name: str,
        *,
        device: str,
        shuffle: bool,
        seed: int,
    ) -> Iterator[torch.Tensor]:
        source = str(cfg.get("cached_activations_source", "local")).lower()

        if source == "local":
            from datasets import load_from_disk  # imported lazily to keep import-time light

            cached_activations_path = Path(cfg["cached_activations_path"])
            ds = load_from_disk(str(cached_activations_path))
            column_names = getattr(ds, "column_names", None) or []
            if hook_name not in column_names:
                raise KeyError(
                    f"Hook column {hook_name!r} not found in cached activations dataset. "
                    f"Available columns: {column_names}"
                )

            # Important: iterator must be effectively infinite because SAELens' ActivationScaler
            # consumes a fixed number of batches (e.g. 1000) up-front for norm estimation.
            epoch = 0
            while True:
                epoch += 1
                epoch_ds = ds
                if shuffle:
                    epoch_ds = ds.shuffle(seed=seed + epoch)

                for row in epoch_ds:
                    act = row[hook_name]
                    t = torch.as_tensor(act, dtype=torch.float32, device=device)  # (context_size, d_in)
                    yield t
            # unreachable

        if source == "hf":
            from datasets import load_dataset

            repo_id = str(cfg["hf_dataset_repo_id"])
            data_dir = cfg.get("hf_data_dir")
            split = str(cfg.get("hf_split", "train"))
            streaming = bool(cfg.get("hf_streaming", True))

            ds = load_dataset(repo_id, data_dir=data_dir, split=split, streaming=streaming)

            if hook_name not in getattr(ds, "column_names", []):
                raise KeyError(
                    f"Hook column {hook_name!r} not found in HF dataset {repo_id!r}. "
                    f"Available columns: {getattr(ds, 'column_names', None)}"
                )

            # Streaming datasets use a shuffle buffer; non-streaming can use full shuffle.
            shuffle_buffer_size = int(cfg.get("hf_shuffle_buffer_size", 10_000))
            epoch = 0
            while True:
                epoch += 1
                epoch_ds = ds
                if shuffle:
                    if streaming:
                        epoch_ds = ds.shuffle(seed=seed + epoch, buffer_size=shuffle_buffer_size)
                    else:
                        epoch_ds = ds.shuffle(seed=seed + epoch)

                for row in epoch_ds:
                    act = row[hook_name]
                    t = torch.as_tensor(act, dtype=torch.float32, device=device)  # (context_size, d_in)
                    yield t
            # unreachable

        raise ValueError(f"Unsupported cached_activations_source={source!r} (expected 'local' or 'hf')")

    def _flatten_token_batches(
        self,
        seq_batches: Iterator[torch.Tensor],
        *,
        d_in: int,
    ) -> Iterator[torch.Tensor]:
        for t in seq_batches:
            if t.ndim != 2 or int(t.shape[-1]) != int(d_in):
                raise ValueError(f"Expected activations shaped (context_size, {d_in}), got {tuple(t.shape)}")
            yield t.reshape(-1, d_in)  # (tokens, d_in)

    async def run(self, data: dict) -> dict:
        if data.get("done"):
            return data

        cfg: dict[str, Any] = data.get("config") or {}

        # Required
        hook_name = str(cfg["hook_name"])
        d_in = int(cfg["d_in"])

        # Optional (smoke-test friendly defaults)
        seed = int(cfg.get("seed", 42))
        device = str(cfg.get("device", "cpu"))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Trainer-level knobs (tokens == samples for resid activations)
        training_tokens = int(cfg.get("training_tokens", 10_000))
        train_batch_size_tokens = int(cfg.get("train_batch_size_tokens", 256))
        lr = float(cfg.get("lr", 3e-4))

        # Data pipeline knobs
        shuffle_dataset = bool(cfg.get("shuffle_dataset", True))
        mixing_buffer_size_tokens = int(cfg.get("mixing_buffer_size_tokens", 50_000))

        # SAE architecture (Standard L1 SAE)
        sae_cfg_dict: dict[str, Any] = cfg.get("sae", {}) or {}
        d_sae = int(sae_cfg_dict.get("d_sae", d_in * 4))
        l1_coefficient = float(sae_cfg_dict.get("l1_coefficient", 5.0))
        normalize_activations = sae_cfg_dict.get("normalize_activations", "expected_average_only_in")
        apply_b_dec_to_input = bool(sae_cfg_dict.get("apply_b_dec_to_input", True))

        # Output + checkpointing
        output_dir = Path(cfg.get("output_dir", "artifacts/sae_lens_runs/smoke"))
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Training SAELens SAE from cached activations: source=%s hook=%s d_in=%s d_sae=%s device=%s tokens=%s batch=%s",
            cfg.get("cached_activations_source", "local"),
            hook_name,
            d_in,
            d_sae,
            device,
            training_tokens,
            train_batch_size_tokens,
        )

        # Build data provider: cached activations -> flattened token batches -> mixing buffer -> fixed batch iterator
        seq_iter = self._activation_iterator_from_cached_dataset(
            cfg,
            hook_name,
            device=device,
            shuffle=shuffle_dataset,
            seed=seed,
        )
        token_iter = self._flatten_token_batches(seq_iter, d_in=d_in)

        from sae_lens.training.mixing_buffer import mixing_buffer

        data_provider = mixing_buffer(
            buffer_size=mixing_buffer_size_tokens,
            batch_size=train_batch_size_tokens,
            activations_loader=token_iter,
        )

        from sae_lens import StandardTrainingSAEConfig
        from sae_lens.config import LoggingConfig, SAETrainerConfig
        from sae_lens.saes.sae import TrainingSAE
        from sae_lens.training.sae_trainer import SAETrainer

        sae_cfg = StandardTrainingSAEConfig(
            d_in=d_in,
            d_sae=d_sae,
            l1_coefficient=l1_coefficient,
            normalize_activations=normalize_activations,
            apply_b_dec_to_input=apply_b_dec_to_input,
        )

        sae = TrainingSAE.from_dict(sae_cfg.to_dict()).to(device)

        trainer_cfg = SAETrainerConfig(
            total_training_samples=training_tokens,
            train_batch_size_samples=train_batch_size_tokens,
            device=device,
            lr=lr,
            lr_end=float(cfg.get("lr_end", lr / 10.0)),
            lr_scheduler_name=str(cfg.get("lr_scheduler_name", "constant")),
            lr_warm_up_steps=int(cfg.get("lr_warm_up_steps", 0)),
            adam_beta1=float(cfg.get("adam_beta1", 0.9)),
            adam_beta2=float(cfg.get("adam_beta2", 0.999)),
            lr_decay_steps=int(cfg.get("lr_decay_steps", 0)),
            n_restart_cycles=int(cfg.get("n_restart_cycles", 1)),
            autocast=bool(cfg.get("autocast", False if device == "cpu" else True)),
            dead_feature_window=int(cfg.get("dead_feature_window", 1000)),
            feature_sampling_window=int(cfg.get("feature_sampling_window", 2000)),
            n_checkpoints=int(cfg.get("n_checkpoints", 0)),
            checkpoint_path=str(cfg.get("checkpoint_path", output_dir / "checkpoints")),
            save_final_checkpoint=bool(cfg.get("save_final_checkpoint", False)),
            logger=LoggingConfig(log_to_wandb=bool(cfg.get("log_to_wandb", False))),
        )

        trainer = SAETrainer(
            cfg=trainer_cfg,
            sae=sae,
            data_provider=data_provider,
            evaluator=None,
        )

        trained_sae = trainer.fit()

        return {
            **data,
            "trained_sae": trained_sae,
            "model_dir": str(output_dir),
            "training_summary": {
                "cached_activations_source": str(cfg.get("cached_activations_source", "local")),
                "cached_activations_path": cfg.get("cached_activations_path"),
                "hf_dataset_repo_id": cfg.get("hf_dataset_repo_id"),
                "hf_data_dir": cfg.get("hf_data_dir"),
                "hf_split": cfg.get("hf_split"),
                "hf_streaming": cfg.get("hf_streaming"),
                "hook_name": hook_name,
                "d_in": d_in,
                "d_sae": d_sae,
                "training_tokens": training_tokens,
                "train_batch_size_tokens": train_batch_size_tokens,
                "device": device,
            },
            "done": False,
        }
