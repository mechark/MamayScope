# Поточний стан пайплайнів лейблінгу

Цей документ описує, як **зараз** працюють пайплайни лейблінгу в проєкті:

1. **Neuron labeling pipeline** (побудова parquet з токенами та fired features).
2. **Feature labeling pipeline** (побудова контекстів для features і LLM-лейблів через OpenRouter).

---

## 1) Neuron labeling pipeline (актуальний entrypoint)

**Entrypoint:** `src/pipelines/neuron_labeling_pipeline.py`  
**Запуск:** `uv run -m src.pipelines.neuron_labeling_pipeline`

Поточний ланцюжок кроків:

`ParquetConversationBatchSource -> MamayActivationProcessor -> SaeFeatureEncodeProcessor -> ParquetNeuronActivationSink`

### 1.1 Джерело: `ParquetConversationBatchSource`

**Файл:** `src/pipelines/sources/parquet_conversation_source.py`

Що робить:
- читає локальний parquet, за замовчуванням `NEURON_LABEL_PROPAGANDA_PARQUET_SOURCE_PATH`
- очікує колонку `conversations` (налаштовується через `NEURON_LABEL_PROPAGANDA_CONVERSATION_COLUMN`)
- кожен елемент `conversations` має формат `[{from, value}, ...]`
- бере всі непорожні `value` у порядку появи та склеює їх у **один текст** через `\n`
- за замовчуванням прибирає **повні дублікати** тексту (`NEURON_LABEL_PROPAGANDA_DEDUP_EXACT=true`)
- повертає батч:
  - `texts: list[str]`
  - `labels: [None, ...]`
  - `done: bool`

Ключові параметри:
- `NEURON_LABEL_PROPAGANDA_PARQUET_SOURCE_PATH`
- `NEURON_LABEL_PROPAGANDA_CONVERSATION_COLUMN`
- `NEURON_LABEL_PROPAGANDA_BATCH_SIZE`
- `NEURON_LABEL_PROPAGANDA_MAX_ROWS`
- `NEURON_LABEL_PROPAGANDA_DEDUP_EXACT`

### 1.2 Отримання активацій: `MamayActivationProcessor`

**Файл:** `src/pipelines/processors/mamay_activation_processor.py`

Що робить:
- відправляє тексти на `MODEL_ENDPOINT` (Mamay API) і отримує активації
- витягує шари `layer_{TARGET_LAYER-1}` і `layer_{TARGET_LAYER}`
- передає далі `output_tensors` для SAE-encode
- ретраїть таймаути/5xx, для проблемних батчів дробить chunk, щоб ізолювати bad rows

### 1.3 Кодування в SAE features: `SaeFeatureEncodeProcessor`

**Файл:** `src/pipelines/processors/sae_feature_encode_processor.py`

Що робить:
- завантажує SAE (SAELens) із налаштувань `SAE_*`
- для кожного токена обчислює список `fired_features` як індекси, де активація `> 0`
- формує записи:
  - `text`
  - `label`
  - `target_layer`
  - `model`
  - `sae_id`
  - `tokens` = список `{token_str, token_id, fired_features}`

### 1.4 Запис: `ParquetNeuronActivationSink`

**Файл:** `src/pipelines/sinks/parquet_neuron_sink.py`

Що робить:
- пише shard-файли в parquet:
  - `data/neurons_labeling_propaganda/labels_batch_000000.parquet`
  - `data/neurons_labeling_propaganda/labels_batch_000001.parquet`
  - ...
- шлях береться з `NEURON_LABEL_PROPAGANDA_OUTPUT_PARQUET_PATH`
- формат рядка: `text`, `label`, `target_layer`, `model`, `sae_id`, `tokens`

---

## 2) Feature labeling pipeline (після побудови neuron parquet)

**Скрипт:** `src/scripts/build_feature_occurrence_index_and_labeler.py`

Типовий потік:
1. Читає parquet-шарди нейронного лейблінгу (`--input-parquet-glob`).
2. Будує зворотній індекс:
   - з `text -> tokens -> fired_features`
   - у `feature_id -> sampled contexts`
3. Для кожної feature збирає контексти (вікно токенів навколо fired token).
4. Ранжує/фільтрує features за якістю контекстів (diversity/entropy гейти).
5. Якщо не `--skip-llm`, викликає OpenRouter для генерації текстового лейблу.
6. Пише JSONL з результатами:
   - за замовчуванням `data/neuron_labels_mamay/results/neuronpedia_feature_labels.jsonl`

---

## 3) OpenRouter сервіс для лейблів features

**Файл:** `src/services/openrouter_labeling_service.py`

Що робить:
- формує prompt-и для назви feature за прикладами контекстів
- викликає OpenRouter `chat/completions`
- парсить JSON-відповідь моделі у структурований `FeatureLabelResult`
- підтримує routing/provider налаштування з `OPENROUTER_*`

---

## 4) Які пайплайни вважати "основними" зараз

- **Основний для даних:** `src/pipelines/neuron_labeling_pipeline.py`  
  (локальний propaganda parquet -> конкатенація діалогу -> Mamay -> SAE -> parquet shards)
- **Основний для інтерпретації/неймінгу features:** `src/scripts/build_feature_occurrence_index_and_labeler.py`  
  (parquet shards -> contexts -> LLM labels -> JSONL)

---

## 5) Швидка перевірка, що все налаштовано

1. Запустити:
   - `uv run -m src.pipelines.neuron_labeling_pipeline`
2. Перевірити, що створюються shard-файли в:
   - `data/neurons_labeling_propaganda/`
3. Запустити feature labeling:
   - `uv run python src/scripts/build_feature_occurrence_index_and_labeler.py --input-parquet-glob "data/neurons_labeling_propaganda/*_batch_*.parquet"`
4. Перевірити JSONL результат:
   - `data/neuron_labels_mamay/results/neuronpedia_feature_labels.jsonl`

