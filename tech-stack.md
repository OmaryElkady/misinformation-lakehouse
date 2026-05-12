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
| transformers | >=4.44.0 | RoBERTa fine-tuning + tokenizer (4.44+ required for numpy 2.x compat; 4.46+ renamed `evaluation_strategy` → `eval_strategy`) |
| torch | Colab pre-installed | Model training backend — do NOT pin in Colab notebook; Colab provides a CUDA-compatible build |
| accelerate | >=0.34.0 | Multi-GPU / mixed-precision support for HuggingFace Trainer (0.34+ required for numpy 2.x compat) |
| datasets | >=2.21.0 | HuggingFace dataset loading (2.21+ required for numpy 2.x compat; older versions carry hard `numpy<2` constraint) |
| scikit-learn | >=1.6.0 | Metrics (f1, precision, recall, confusion matrix) — training + tests |
| matplotlib | 3.9.0 | Confusion matrix PNG artifact logged to MLflow |
| mlflow | >=2.15.0 | Experiment tracking + model registry — client and Docker server must both be >=2.15.0; 2.15 added host header validation, CSRF protection, and the logged-models endpoint |
| groq | 0.9.0 | Llama 3 inference (free tier) |
| prefect | >=3.0,<4.0 | Pipeline orchestration (3.x required for Python 3.12) |
| fastapi | 0.111.0 | Inference API server |
| uvicorn | 0.30.1 | ASGI server for FastAPI |
| pydantic | 2.7.1 | Data validation |
| pydantic-settings | 2.3.0 | Settings management from env vars |
| boto3 | 1.34.120 | AWS S3 access |
| python-dotenv | 1.0.1 | .env file loading |
| loguru | 0.7.2 | Structured logging |

> **setuptools compatibility:** MLflow calls `import pkg_resources` at import time (via CLI and some internals).
> `setuptools >= 80` removed `pkg_resources` as a standalone module, which breaks `mlflow server`
> and prevents pytest from collecting serving tests. `setuptools<80` is pinned in `requirements.txt`.
> MLflow imports in `src/serving/` are kept lazy (inside functions) as an additional guard.

> **Colab-specific versions:** The notebook (`notebooks/colab_training.ipynb`) uses minimum-version
> bounds (`>=`) rather than exact pins for ML packages. This is intentional — Colab's pre-installed
> CUDA torch must not be overridden, and packages must be post-numpy-2.0 releases. Do not backport
> the notebook's package list to `requirements.txt` or vice versa.

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
