# Architecture

> Load this when making structural decisions, adding new modules, or understanding how the system fits together.

---

## System Overview

End-to-end misinformation detection platform built on a lakehouse architecture.
Data flows from public datasets (LIAR, FakeNewsNet) via HuggingFace Datasets into
Bronze Delta tables, gets cleaned and feature-engineered through Silver and Gold layers
via PySpark, and feeds a fine-tuned RoBERTa classifier tracked in MLflow.
Predictions are served in real-time via a Dockerized FastAPI endpoint that also
calls Groq's Llama 3 API for human-readable explanations.

---

## Data Flow

```
LIAR Dataset ────┐
                 ├──► Bronze (raw)
FakeNewsNet ─────┘    Delta Table
(via HuggingFace)          │
                           ▼
                    Silver (clean)
                    Delta Table
                    - deduped
                    - normalized labels (0/1)
                    - basic text features
                           │
                           ▼
                     Gold (features)
                     Delta Table
                     - TF-IDF features
                     - sentiment scores
                     - model predictions
                           │
                    ┌──────┴──────┐
                    ▼             ▼
             RoBERTa          Groq Llama 3
             Fine-tune        (explanation)
             MLflow ──────► Model Registry
                                  │
                                  ▼
                           FastAPI /predict
                           Docker container
```

---

## Key Modules

| Module | Location | Responsibility |
|--------|----------|----------------|
| Config | `src/config.py` | Single source of truth for all env vars |
| Spark factory | `src/spark_session.py` | Consistent SparkSession with Delta + S3 config |
| Static ingestion | `src/ingestion/ingest_static.py` | Load LIAR/FakeNewsNet via HuggingFace → Bronze Delta |
| Bronze→Silver | `src/processing/bronze_to_silver.py` | Clean, deduplicate, normalize labels |
| Silver→Gold | `src/processing/silver_to_gold.py` | Feature engineering, sentiment, TF-IDF |
| Training — export | `src/training/train.py::export_gold_to_parquet` | Read Gold Delta → write train/val parquets locally |
| Training — fine-tune | `src/training/train.py::train` | Run on Colab; reads parquets, fine-tunes, logs to MLflow via ngrok |
| Training — registry | `src/training/train.py::register_model` | Run locally after Colab; registers + promotes model |
| API server | `src/serving/app.py` | FastAPI inference endpoint |
| Pipeline | `src/orchestration/pipeline.py` | Prefect flow wiring all steps |
| Model gate | `scripts/validate_model.py` | CI quality threshold check |

---

## Patterns in Use

- **Medallion architecture** — Bronze (raw) → Silver (clean) → Gold (features). Never skip a layer.
- **Config singleton** — `settings` from `src/config.py` is imported everywhere. No inline env reads.
- **Spark factory** — `get_spark()` is called once per job. Never build a session inline.
- **Storage abstraction** — `settings.delta_path("bronze/silver/gold")` resolves to local or S3 based on `STORAGE_MODE`. All path references go through this method.
- **MLflow lifecycle** — every training run logs params, metrics, and artifacts. Model is registered and promoted via the registry, never loaded from a file path directly.
- **Two-phase training** — GPU training runs on Google Colab, not locally. Phase 1 (local): `export_gold_to_parquet()` writes `data/exports/train.parquet` + `val.parquet`. Phase 2 (Colab): `notebooks/colab_training.ipynb` reads those parquets, fine-tunes, and logs to the local MLflow server via an ngrok tunnel. Phase 3 (local): `register_model(run_id)` promotes the result. See `scripts/setup_colab.md` for the full walkthrough.

---

## Boundaries: What NOT to Cross

- `src/serving/` must not import PySpark — the API container doesn't have it
- `src/training/` reads from Gold Delta table only — never from raw files or Silver
- Route handlers in `src/serving/app.py` must not contain ML logic — call a loader/predictor class
- Tests in `tests/unit/` must not import from `src/spark_session.py` or any PySpark module
