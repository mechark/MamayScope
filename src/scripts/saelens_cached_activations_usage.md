## SAELens cached activations: how to use the exported dataset

### What the builder script creates

`src/scripts/build_saelens_cached_activations_dataset.py` writes a Hugging Face `datasets` directory (via `Dataset.save_to_disk()`).

- **hook column**: `blocks.33.hook_resid_post` (or whatever you pass as `--hook_name`)
- **shape per row**: `(context_size, d_in)` (defaults: `1024 x 3584`)
- **dtype**: `float32`
- **token_ids**: not included (your Parquet batches don’t contain them)

SAELens validates this when loading cached activations (it checks the hook column exists and its feature shape equals `(context_size, d_in)`). See SAELens API docs (source shown on that page): `https://decoderesearch.github.io/SAELens/dev/api/#sae_lens.CacheActivationsRunnerConfig` → `ActivationsStore.load_cached_activation_dataset()`.

### How to train with cached activations in SAELens

In your SAELens training config:

- **`use_cached_activations=True`**
- **`cached_activations_path="/path/to/the/dataset_dir"`** (the directory you wrote with `save_to_disk`)
- **`hook_name="blocks.33.hook_resid_post"`** (must match the dataset column name)
- **`context_size=1024`** (must match the dataset feature shape)
- **`d_in=3584`** (must match the dataset feature shape / your model hidden size)

SAELens training docs on caching activations: `https://decoderesearch.github.io/SAELens/dev/training_saes/#caching-activations`.

### Padding stats

The builder prints these at the end:

- `padded_rows`: `0` or `1`
- `padded_tokens`: number of zero-vectors appended to fill the final incomplete window
- `padded_token_fraction`: `padded_tokens / (total_real_tokens + padded_tokens)`

Because we pack tokens across all sequences, padding should be **at most `context_size-1` tokens total** across the entire dataset.

