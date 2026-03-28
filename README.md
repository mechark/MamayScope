# MamayScope

**Перший інструмент механістичної інтерпретованості для українських великих мовних моделей (проект у розробці...)**

MamayScope — це дослідницька платформа для аналізу латентних просторів моделі [MamayLM](https://huggingface.co/INSAIT-Institute/MamayLM-Qwen2.5-7B-v0.2) за допомогою розріджених автоенкодерів (Sparse Autoencoders). Система дозволяє витягувати активації residual потоку, тренувати SAE, знаходити інтерпретовані фічі та автоматично розмічати їх через LLM (OpenRouter).

Архітектура системи MamayScope

---

## Навіщо це потрібно

Великі мовні моделі — чорні скриньки. Вони генерують текст, але ми не розуміємо *як* і *чому*. Механістична інтерпретованість — це напрямок, який досліджує внутрішні механізми нейромереж на рівні окремих компонентів.

**Проблема:** на момент створення MamayScope не існувало жодного інструменту механістичної інтерпретованості для українських LLM. Жодних натренованих SAE, жодних словників фіч, жодних адаптованих пайплайнів.

**Рішення:** MamayScope — end-to-end система, що покриває весь цикл: від збору активацій до інтерактивного перегляду розмічених фіч.

---

## Що всередині

### vLLM-MamayHook

Плагін-бібліотека для [vLLM](https://github.com/IBM/vLLM-Hook), що дозволяє інспектувати внутрішні стани трансформера під час інференсу:

- **ResidualStreamWorker** — перехоплює залишковий потік через PyTorch forward hooks
- **ProbeHookQKWorker** — захоплює Q/K вектори з голів уваги
- **SteerActivationWorker** — модифікує активації для activation steering
- **PluginRegistry** — модульна архітектура з реєстром компонентів

Комунікація між процесами відбувається через змінні середовища (`VLLM_HOOK_DIR`, `VLLM_HOOK_FLAG`, `VLLM_RUN_ID`), оскільки воркери vLLM працюють у окремих підпроцесах.

### MamayScope Зараз

Пайплайн-фреймворк, побудований на композиції асинхронних кроків `PipelineStep`:

```
Source → Processor → Processor → Sink
```

Кожен крок приймає `dict`, трансформує його та передає далі. `PipelineExecutor` оркеструє послідовне виконання.

---

## Ключові пайплайни

### Тренування SAE

```
CachedActivationsConfig → SAELensTrainer → ModelFileSink → HuggingFaceHubSink
```

- Словник: **16 384** фічі, вхідна розмірність: **3 584**
- Hook point: `blocks.33.hook_resid_post` (residual потік шару 33)
- L1-регуляризація для розрідженості
- Інтеграція з Weights & Biases для логування
- Автоматичний push на [Hugging Face Hub](https://huggingface.co/mechark/MamaySAE)

### Розмітка нейронів

Пайплайн розмітки нейронів

```
ParquetConversationSource → MamayActivationProcessor → SaeFeatureEncodeProcessor → FeatureOccurrenceIndex → OpenRouterLabeling
```

- HTTP-запити до vLLM-MamayHook API для отримання активацій
- Кодування через SAE з вирівнюванням токенізації
- Побудова індексу входжень фіч
- Автоматична анотація через LLM (OpenRouter API)

### Формування корпусу

Наразі корпус акцинтований на політичну тематику, проте має бути розширений до всіх тем.

```
LAPA HuggingFace → ParquetTextBatchSource → UkrainianPoliticalKeywordFilter → TextCorpusParquetSink
```

- 200+ ключових слів для фільтрації політичного дискурсу
- NFC-нормалізація та case-folding
- Шардований Parquet-вихід

---

## Результати на зараз

### Натренована SAE-модель


| Метрика                  | Значення     |
| ------------------------ | ------------ |
| Explained Variance       | 0.71         |
| Середня розрідженість L0 | ~175         |
| MSE Loss                 | 6500 → 1500  |
| Навчальний корпус        | 500K токенів |
| Розмір словника          | 16 384 фічі  |


Модель опублікована: `[mechark/MamaySAE](https://huggingface.co/mechark/MamaySAE)`

### Виявлені фічі

Аналіз 205 розмічених фіч виявив семантично значущі концепції:

Розподіл фіч за категоріями

**Приклади:**

- **Feature 2569** — «Геополітичний дискурс та міжнародне право» (активується на текстах про суверенітет, міжнародні договори, територіальну цілісність)
- **Feature 1115** — «Російські пропагандистські наративи» (виділяє маніпулятивну риторику, whataboutism, дезінформаційні кліше)
- **Feature 6606** — «Дискурс причин та наслідків війни 2014 року» (фокус на хронології подій Майдану, анексії Криму)
- **Feature 1621** — «Хронологія та документальна верифікація» (дати, номери резолюцій, посилання на офіційні документи)

### Feature Browser

Інтерфейс перегляду фіч

---

## Швидкий старт

```bash
# Клонування
git clone <repo-url>
cd MamayScope

# Встановлення залежностей (потрібен uv)
uv sync

# Запуск API сервера для збору активацій
uv run -m src.api.main

# Тренування SAE (smoke test)
uv run -m src.pipelines.saelens_sae_training_pipeline \
  --config configs/saelens_sae_trainer_smoke.yaml

# Повне тренування SAE
uv run -m src.pipelines.saelens_sae_training_pipeline \
  --config configs/saelens_sae_trainer_full.yaml

# Пайплайн розмітки нейронів
uv run -m src.pipelines.neuron_labeling_pipeline

# Побудова індексу фіч та автоматична розмітка
uv run -m src.scripts.build_feature_occurrence_index_and_labeler \
  --skip-llm --max-rows 1000

# Генерація HTML-браузера фіч
uv run -m src.scripts.build_feature_label_browser
```

---

## Конфігурація

Основні змінні середовища:


| Змінна              | Опис                        | Default                 |
| ------------------- | --------------------------- | ----------------------- |
| `MODEL_ENDPOINT`    | URL до Mamay API            | `http://localhost:8000` |
| `TARGET_LAYER`      | Шар для збору активацій     | `33`                    |
| `SAE_HF_REPO_ID`    | SAE-модель на HF Hub        | `mechark/MamaySAE`      |
| `SAE_HF_REVISION`   | Ревізія для відтворюваності | —                       |
| `BATCH_SIZE`        | Розмір батчу                | —                       |
| `PARALLEL_REQUESTS` | Паралельні запити до API    | —                       |


---

## Структура проєкту

```
MamayScope/
├── src/
│   ├── api/                # FastAPI — ендпоінт /activations
│   ├── core/               # Налаштування
│   ├── pipelines/
│   │   ├── base.py         # Абстрактний PipelineStep
│   │   ├── pipeline_executor.py
│   │   ├── sources/        # Джерела даних (Parquet, HF, YAML)
│   │   ├── processors/     # Активації, SAE-кодування
│   │   └── sinks/          # Parquet, JSONL, HF Hub
│   ├── scripts/            # CLI-утиліти
│   ├── services/           # HookedMamayService, OpenRouter
│   └── schemas/            # Pydantic-моделі
├── configs/                # YAML-конфігурації тренування
├── docs/                   # Документація (українською)
└── tests/
```

---

## Ліцензія

MIT