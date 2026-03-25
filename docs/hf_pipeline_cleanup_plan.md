## Follow-up cleanup plan (HF-related pipeline removal)

You asked: “remove everything else from the pipeline that uses HF”.

This repo currently has two separate HF-related concepts:

- **HF dataset access** (loading cached activations from the Hub via `datasets.load_dataset`)
- **HF model upload** (pushing a trained SAE model directory to the Hub via `huggingface_hub`)

Depending on what you want to keep, here are safe cleanup options.

### Option A: Remove only HF dataset loading (keep optional model upload)

- **Revert smoke config** back to local dataset path:
  - `configs/saelens_sae_trainer_smoke.yaml`: remove `cached_activations_source: hf` and HF dataset fields; set `cached_activations_source: local` and `cached_activations_path: ...`
- **Simplify trainer**:
  - `src/pipelines/trainers/sae_trainer.py`: remove the `source == "hf"` branch and all `hf_*` config parsing.

### Option B: Remove only HF model upload (keep HF dataset loading)

- Delete `src/pipelines/sinks/huggingface_hub_model_sink.py`
- Update `src/pipelines/sinks/__init__.py` to stop exporting it
- Update `src/pipelines/saelens_sae_training_pipeline.py` to stop wiring the HF sink step
- Remove HF upload keys from YAML configs (`push_to_hub`, `hf_repo_id`, etc.)

### Option C: Remove all HF usage from the pipeline

Do both Option A and Option B.

### What stays (recommended)

Even if you remove HF from the *pipeline*, it’s still useful to keep standalone scripts:
- `src/scripts/download_mamay_sae_dataset.py` for downloading datasets onto pods
- `src/scripts/push_cached_activations_to_hub.py` for publishing datasets (optional)

