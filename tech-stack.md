# Tech Stack

> Load this when adding dependencies, setting up CI, or making tooling decisions.
> Pinned versions prevent Claude from suggesting outdated or incompatible packages.

---

## Runtime & Language

| Thing | Version |
|-------|---------|
| Python | 3.12 |
| WSL2 | Ubuntu 22.04 |
| Java (for Spark) | 17 (OpenJDK) |

---

## Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pyspark | 4.1.1 | Distributed data processing |
| delta-spark | 4.1.0 | Delta Lake on local/S3 — requires `configure_spark_with_delta_pip(builder)` to wire JARs from Maven |
| pandas | 2.2.2 | Small data, ingestion helpers |
| pyarrow | 15.0.2 | Parquet / Delta serialization |
| transformers | 4.41.2 | RoBERTa fine-tuning + tokenizer |
| torch | 2.3.0 | Model training backend |
| accelerate | 0.30.0 | Multi-GPU / mixed-precision support for HuggingFace Trainer |
| datasets | 2.19.2 | HuggingFace dataset loading |
| scikit-learn | 1.5.0 | Metrics (f1, precision, recall, confusion matrix) — training + tests |
| matplotlib | 3.9.0 | Confusion matrix PNG artifact logged to MLflow |
| mlflow | 2.13.0 | Experiment tracking + model registry |
| groq | 0.9.0 | Llama 3 inference (free tier) |
| prefect | >=3.0,<4.0 | Pipeline orchestration (3.x required for Python 3.12) |
| fastapi | 0.111.0 | Inference API server |
| uvicorn | 0.30.1 | ASGI server for FastAPI |
| pydantic | 2.7.1 | Data validation |
| pydantic-settings | 2.3.0 | Settings management from env vars |
| boto3 | 1.34.120 | AWS S3 access |
| python-dotenv | 1.0.1 | .env file loading |
| loguru | 0.7.2 | Structured logging |

> **setuptools compatibility:** `mlflow==2.13.0` calls `import pkg_resources` at import time.
> `setuptools >= 80` removed `pkg_resources` as a standalone module, which breaks the mlflow
> import and prevents pytest from collecting serving tests. Pin `setuptools<80` in the venv, or
> keep mlflow imports lazy (inside functions) in `src/serving/` — which is what the codebase does.

> **Prefect + Python 3.12:** Prefect 2.x (including 2.19.5) does not run on Python 3.12 because
> `pydantic.v1` (bundled in pydantic 2.x) calls `ForwardRef._evaluate()` with a positional
> `set()` argument that was removed in Python 3.12. Use `prefect>=3.0,<4.0` which ships with
> native pydantic v2 support. The Docker image is `prefecthq/prefect:3-python3.12`.

---

## Approved Libraries

- **Logging:** `loguru` — not `logging`, not `print()`
- **HTTP client:** `httpx` — not `requests`, not `urllib`
- **Data validation:** `pydantic` v2 — not dataclasses for API models
- **Config:** `pydantic-settings` via `src/config.py` — not `dynaconf`, not raw `os.getenv()`
- **Testing:** `pytest` + `pytest-asyncio` + `pytest-cov`
- **Retry logic:** `tenacity` — not manual retry loops

---

## Off-Limits Libraries

- `requests` — use `httpx` instead
- `logging` module directly — use `loguru`
- `flask` — we use `fastapi`
- `airflow` — we use `prefect`
- `scikit-learn` pipelines for the main model — use HuggingFace `Trainer`

---

## Dev Tooling

| Tool | Config file | Purpose |
|------|------------|---------|
| ruff | `pyproject.toml` | Linting |
| black | `pyproject.toml` | Formatting |
| pytest | `pyproject.toml` | Testing |
| GitHub Actions | `.github/workflows/ci.yml` | CI pipeline |

---

## CI/CD Order

1. Lint: `ruff check src/ tests/`
2. Format check: `black --check src/ tests/`
3. Unit tests: `pytest tests/unit/ -m unit`
4. Docker build + health check
5. Model validation gate (main branch only)
