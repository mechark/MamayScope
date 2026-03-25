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

    def _as_float_coeffs(self, coeffs: dict[str, object]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in coeffs.items():
            # SAELens sometimes stores TrainCoefficientConfig-like values; for eval we only need floats.
            if hasattr(v, "value"):
                out[k] = float(getattr(v, "value"))
            else:
                out[k] = float(v)  # type: ignore[arg-type]
        return out

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
        validation_tokens = int(cfg.get("validation_tokens", 0))
        validation_n_eval_batches = int(cfg.get("validation_n_eval_batches", 20))

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
        from sae_lens.saes.sae import TrainStepInput

        sae_cfg = StandardTrainingSAEConfig(
            d_in=d_in,
            d_sae=d_sae,
            l1_coefficient=l1_coefficient,
            normalize_activations=normalize_activations,
            apply_b_dec_to_input=apply_b_dec_to_input,
        )

        sae = TrainingSAE.from_dict(sae_cfg.to_dict()).to(device)

        # Optional validation split/evaluator
        evaluator = None
        if validation_tokens > 0:
            # We split by activation rows. Each dataset row contributes `context_size` tokens.
            # Our cached activations are (context_size, d_in), so row_tokens is its 0th dim.
            # We assume context_size is consistent (1024 in your dataset).
            val_rows = max(1, validation_tokens // 1024)

            source = str(cfg.get("cached_activations_source", "local")).lower()
            streaming = bool(cfg.get("hf_streaming", True)) if source == "hf" else False
            if source == "hf" and streaming:
                raise ValueError(
                    "Validation split requires non-streaming dataset. "
                    "Set hf_streaming: false or use cached_activations_source: local."
                )

            # Load dataset once, then split deterministically.
            if source == "local":
                from datasets import load_from_disk

                ds = load_from_disk(str(Path(cfg["cached_activations_path"])))
            elif source == "hf":
                from datasets import load_dataset

                ds = load_dataset(
                    str(cfg["hf_dataset_repo_id"]),
                    data_dir=cfg.get("hf_data_dir"),
                    split=str(cfg.get("hf_split", "train")),
                    streaming=False,
                )
            else:
                ds = None

            if ds is None:
                raise ValueError(f"Unsupported cached_activations_source={source!r}")

            if hook_name not in getattr(ds, "column_names", []):
                raise KeyError(
                    f"Hook column {hook_name!r} not found in dataset. Available columns: {getattr(ds, 'column_names', None)}"
                )

            n_rows = len(ds)  # non-streaming only
            if val_rows >= n_rows:
                raise ValueError(f"validation split too large: val_rows={val_rows} >= n_rows={n_rows}")

            val_ds = ds.select(range(0, val_rows))
            train_ds = ds.select(range(val_rows, n_rows))

            def _seq_iter_from_ds(split_ds):
                epoch = 0
                while True:
                    epoch += 1
                    epoch_ds = split_ds
                    if shuffle_dataset:
                        epoch_ds = epoch_ds.shuffle(seed=seed + epoch)
                    for row in epoch_ds:
                        act = row[hook_name]
                        yield torch.as_tensor(act, dtype=torch.float32, device=device)

            val_seq_iter = _seq_iter_from_ds(val_ds)
            val_token_iter = self._flatten_token_batches(val_seq_iter, d_in=d_in)
            val_data_provider = mixing_buffer(
                buffer_size=mixing_buffer_size_tokens,
                batch_size=train_batch_size_tokens,
                activations_loader=val_token_iter,
            )

            def _evaluator(sae_model, _train_provider, activation_scaler):
                sae_model.eval()
                coeffs = self._as_float_coeffs(sae_model.get_coefficients())
                total_mse = 0.0
                total_explained_var = 0.0
                total_l0 = 0.0

                for _ in range(validation_n_eval_batches):
                    batch = next(val_data_provider).to(sae_model.device)
                    scaled = activation_scaler(batch)
                    out = sae_model.training_forward_pass(
                        TrainStepInput(
                            sae_in=scaled,
                            coefficients=coeffs,
                            dead_neuron_mask=None,
                            n_training_steps=0,
                            is_logging_step=False,
                        )
                    )
                    mse = float(out.losses["mse_loss"].item())
                    total_mse += mse

                    feature_acts = out.feature_acts
                    l0 = feature_acts.bool().float().sum(-1).to_dense().mean().item()
                    total_l0 += float(l0)

                    sae_in = out.sae_in
                    sae_out = out.sae_out
                    per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
                    total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
                    explained_variance = 1 - per_token_l2_loss.mean() / total_variance.mean()
                    total_explained_var += float(explained_variance.item())

                n = float(validation_n_eval_batches)
                return {
                    "val/loss_mse": total_mse / n,
                    "val/metrics_explained_variance": total_explained_var / n,
                    "val/metrics_l0": total_l0 / n,
                    "val/val_rows": int(val_rows),
                }

            evaluator = _evaluator

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
            logger=LoggingConfig(
                log_to_wandb=bool(cfg.get("log_to_wandb", False)),
                wandb_project=str(cfg.get("wandb_project", "mamayscope-sae")),
                wandb_entity=(str(cfg["wandb_entity"]) if cfg.get("wandb_entity") else None),
                wandb_id=(str(cfg["wandb_id"]) if cfg.get("wandb_id") else None),
                run_name=(str(cfg["wandb_run_name"]) if cfg.get("wandb_run_name") else None),
                wandb_log_frequency=int(cfg.get("wandb_log_frequency", 10)),
                eval_every_n_wandb_logs=int(cfg.get("eval_every_n_wandb_logs", 100)),
            ),
        )

        trainer = SAETrainer(
            cfg=trainer_cfg,
            sae=sae,
            data_provider=data_provider,
            evaluator=evaluator,
        )

        # SAELens' SAETrainer will call wandb.log(...) directly when log_to_wandb is enabled,
        # so we must initialize wandb in *this* process before training starts.
        wandb_run = None
        if trainer_cfg.logger.log_to_wandb:
            import wandb

            wandb_run = wandb.init(
                project=trainer_cfg.logger.wandb_project,
                entity=trainer_cfg.logger.wandb_entity,
                id=trainer_cfg.logger.wandb_id,
                name=trainer_cfg.logger.run_name,
                resume="allow",
                config={
                    "hook_name": hook_name,
                    "d_in": d_in,
                    "d_sae": d_sae,
                    "training_tokens": training_tokens,
                    "train_batch_size_tokens": train_batch_size_tokens,
                    "device": device,
                    "validation_tokens": validation_tokens,
                    "cached_activations_source": cfg.get("cached_activations_source", "local"),
                },
            )

        try:
            trained_sae = trainer.fit()
        finally:
            if wandb_run is not None:
                try:
                    import wandb

                    wandb.finish()
                except Exception:
                    # Don't fail training completion due to logging teardown issues.
                    self.logger.exception("wandb.finish() failed")

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
