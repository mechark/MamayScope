# `src/scripts` Usage Guide

This file explains what each script in `src/scripts` does and how to run it.

General run pattern from repo root:

```bash
uv run -m src.scripts.<script_name_without_py> [args...]
```

---

## 1) `build_feature_occurrence_index_and_labeler.py`

Purpose:
- Scans neuron-label Parquet batches.
- Builds feature -> sampled context windows.
- Optionally calls OpenRouter to generate labels and writes JSONL.

Typical commands:

```bash
# Index only (no LLM calls), quick smoke run
uv run -m src.scripts.build_feature_occurrence_index_and_labeler \
  --skip-llm \
  --max-rows 1000
```

```bash
# Full labeling run with resume support
uv run -m src.scripts.build_feature_occurrence_index_and_labeler \
  --input-parquet-glob "data/neurons_labeling_propaganda/*_batch_*.parquet" \
  --output-jsonl "data/neurons_labeling_propaganda/results/neuronpedia_feature_labels.jsonl" \
  --top-k 20 \
  --min-occurrences 20 \
  --resume
```

Useful flags:
- `--skip-llm`: index only.
- `--resume` / `--skip-existing`: skip already labeled features in output.
- `--openrouter-model`: override model.
- `--max-features-to-label`: cap labeling workload.

---

## 2) `build_feature_label_browser.py`

Purpose:
- Builds a static HTML browser for an existing feature-label JSONL file.
- No model inference; pure browsing UI.

Typical commands:

```bash
# Use defaults
uv run -m src.scripts.build_feature_label_browser
```

```bash
# Custom input/output
uv run -m src.scripts.build_feature_label_browser \
  --input-jsonl "data/neuron_labels_mamay/results/neuronpedia_feature_labels.jsonl" \
  --output-html "data/neuron_labels_mamay/results/feature_label_browser.html" \
  --max-rows 5000
```

---

## 3) `build_sae_vis_from_parquet.py`

Purpose:
- Reads token IDs from neuron-label Parquet batches.
- Loads SAE + TransformerLens model and renders local SAE-Vis HTML.

Typical commands:

```bash
# Validate setup only (no HTML render)
uv run -m src.scripts.build_sae_vis_from_parquet \
  --dry-run \
  --feature-ids 700 \
  --max-rows 128
```

```bash
# Render dashboard from feature IDs in JSONL
uv run -m src.scripts.build_sae_vis_from_parquet \
  --feature-ids-file "data/neuron_labels_mamay/results/neuronpedia_feature_labels.jsonl" \
  --max-features 8
```

```bash
# Render one file per feature
uv run -m src.scripts.build_sae_vis_from_parquet \
  --feature-ids "700,2569,2640" \
  --save-per-feature
```

Useful flags:
- `--parquet-glob`: source parquet files.
- `--feature-ids` / `--feature-ids-file`: select features.
- `--device`: `auto|cuda|mps|cpu`.
- `--tl-model-name`: optional TransformerLens model override.

---

## 4) `build_saelens_cached_activations_dataset.py`

Purpose:
- Converts MamayScope activation Parquet into SAELens cached-activations dataset format.
- Writes a local `datasets.save_to_disk()` directory.

Typical commands:

```bash
uv run -m src.scripts.build_saelens_cached_activations_dataset \
  --parquet_glob "data/*batch_*.parquet" \
  --hook_name "blocks.33.hook_resid_post" \
  --context_size 1024 \
  --d_in 3584 \
  --out_dir "saelens_cached_activations_blocks33_resid_post"
```

```bash
# Smaller test build
uv run -m src.scripts.build_saelens_cached_activations_dataset \
  --parquet_glob "data/*batch_*.parquet" \
  --context_size 128 \
  --shard_rows 64 \
  --log_every_rows 10 \
  --out_dir "tmp_cached_activations_test"
```

Useful flags:
- `--parquet_column` (default `output_tensor`).
- `--allow_non_batch` to include non-`batch_` parquet files.
- `--shard_rows` to control per-shard size.

---

## 5) `push_cached_activations_to_hub.py`

Purpose:
- Uploads a local `datasets.save_to_disk()` dataset directory to Hugging Face Hub.

Prerequisite:
- Authenticate (`HF_TOKEN` env var or `huggingface-cli login`).

Typical command:

```bash
uv run -m src.scripts.push_cached_activations_to_hub \
  --dataset_dir "saelens_cached_activations_blocks33_resid_post" \
  --repo_id "YOUR_USERNAME/YOUR_DATASET_NAME" \
  --revision "main" \
  --commit_message "Add SAELens cached activations dataset"
```

Optional:
- Add `--private` for private repo.
- Add `--num_shards N` to control upload sharding.

---

## 6) `download_mamay_sae_dataset.py`

Purpose:
- Downloads Mamay SAE dataset from HF Hub and saves it locally with `save_to_disk`.

Prerequisite:
- Set token:

```bash
export HF_TOKEN="your_token_here"
```

Typical command:

```bash
uv run -m src.scripts.download_mamay_sae_dataset \
  --repo_id "mechark/MamaySAEDataset" \
  --data_dir "data" \
  --split "train" \
  --out_dir "MamaySAEDataset" \
  --hook_name "blocks.33.hook_resid_post"
```

---

## Quick sequence (end-to-end)

```bash
# 1) Build/refresh index and labels
uv run -m src.scripts.build_feature_occurrence_index_and_labeler --resume

# 2) Build static browser
uv run -m src.scripts.build_feature_label_browser

# 3) Build SAE-Vis HTML
uv run -m src.scripts.build_sae_vis_from_parquet --max-features 8
```
