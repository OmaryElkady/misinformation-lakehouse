# Misinformation Detection Lakehouse

> End-to-end MLOps platform that ingests Reddit posts and public datasets,
> processes them through a Bronze→Silver→Gold Delta Lake pipeline, fine-tunes
> RoBERTa for misinformation classification, and serves predictions via FastAPI.

---

## Quick Start

```bash
# Install (in WSL2, Python 3.12)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start services (MLflow + Prefect + API)
docker compose up -d

# Test
pytest tests/unit/ -v
```

---

## Project Structure

```
src/
  ingestion/       # LIAR dataset + FakeNewsNet → Bronze Delta table
  processing/      # Bronze → Silver → Gold PySpark jobs
  training/        # RoBERTa fine-tune + MLflow experiment tracking
  serving/         # FastAPI inference server
  orchestration/   # Prefect pipeline flows
  config.py        # ALL env vars loaded here — never use os.getenv() elsewhere
  spark_session.py # SparkSession factory — always use get_spark(), never inline
tests/
  unit/            # Fast, no Spark/network — these run in CI
  integration/     # Require Spark locally — skip in CI with @pytest.mark.skip
```

---

## Key Conventions

- **Language:** Python 3.12
- **Test framework:** pytest — unit tests in `tests/unit/`, integration in `tests/integration/`
- **Formatter:** ruff + black — auto-runs via `.claude/hooks/auto-format.sh`
- **Config:** pydantic-settings via `src/config.py` — always import `settings` from there

---

## Data Sources

- **LIAR dataset** and **FakeNewsNet** loaded via HuggingFace Datasets library — no API key needed
- Reddit ingestion was descoped — HuggingFace streaming is used for live ingestion instead

---

## Non-Obvious Things

- `STORAGE_MODE=local` for dev/CI, `STORAGE_MODE=s3` for production branch
- Integration tests use `@pytest.mark.skip` — never run in CI, run manually only
- The Gold Delta table is the feature store — all training reads from there, not raw files
- MLflow model registry name is always `"misinformation-roberta-v1"`
- Never call `SparkSession.builder` inline — always use `get_spark()` from `src/spark_session.py`
- Never call `os.getenv()` anywhere — all config goes through `src/config.py`
- Training is three-phase (no local GPU): (1) run `export_gold_to_parquet()` locally, (2) run `notebooks/colab_training.ipynb` on Colab with an ngrok tunnel to local MLflow, (3) run `register_model(run_id)` locally — see `scripts/setup_colab.md`
- Model is promoted to Production only if `eval_f1 >= 0.80`; otherwise it lands in Staging
- `ModelLoader.load()` never raises — it catches all exceptions and logs them, leaving `loaded=False`; the app always starts even when MLflow is unreachable
- The serving layer uses a lifespan context manager (`@asynccontextmanager async def lifespan`) — never `@app.on_event`, which is deprecated in FastAPI 0.111
- mlflow is imported lazily inside `ModelLoader.load()` (not at module level) — this avoids import failures in envs where `setuptools >= 80` has removed `pkg_resources`
- `groq` and `datasets` are also optional at import time — both are wrapped in `try/except ImportError` at module level so CI's minimal install doesn't crash; the fallback `= None` keeps the name patchable in tests
- The serving Dockerfile must include `pydantic-settings` — `src/config.py` imports it at module level and the container won't start without it

---

## Steering Files

- `gotchas.md` — mistakes to avoid in this codebase (includes PySpark mock traps)
- `architecture.md` — data flow and module responsibilities
- `tech-stack.md` — pinned versions and approved libraries
- `scripts/setup_colab.md` — step-by-step Colab training walkthrough


