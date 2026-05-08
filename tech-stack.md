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
| delta-spark | 3.3.0 | Delta Lake on local/S3 |
| pandas | 2.2.2 | Small data, ingestion helpers |
| pyarrow | 16.0.0 | Parquet / Delta serialization |
| transformers | 4.41.2 | RoBERTa fine-tuning + tokenizer |
| torch | 2.3.0 | Model training backend |
| datasets | 2.19.2 | HuggingFace dataset loading |
| mlflow | 2.13.0 | Experiment tracking + model registry |
| groq | 0.9.0 | Llama 3 inference (free tier) |
| prefect | 2.19.5 | Pipeline orchestration |
| fastapi | 0.111.0 | Inference API server |
| uvicorn | 0.30.1 | ASGI server for FastAPI |
| pydantic | 2.7.1 | Data validation |
| pydantic-settings | 2.3.0 | Settings management from env vars |
| praw | 7.7.1 | Reddit API client |
| boto3 | 1.34.120 | AWS S3 access |
| python-dotenv | 1.0.1 | .env file loading |
| loguru | 0.7.2 | Structured logging |

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
