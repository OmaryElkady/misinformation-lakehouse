# Misinformation Detection Lakehouse

> End-to-end MLOps platform that ingests Reddit posts and public datasets,
> processes them through a Bronze→Silver→Gold Delta Lake pipeline, fine-tunes
> RoBERTa for misinformation classification, and serves predictions via FastAPI.

---

## Quick Start

```bash
# Install (in WSL2, Python 3.11)
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
  ingestion/       # LIAR dataset + Reddit API → Bronze Delta table
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

- **Language:** Python 3.11
- **Test framework:** pytest — unit tests in `tests/unit/`, integration in `tests/integration/`
- **Formatter:** ruff + black — auto-runs via `.claude/hooks/auto-format.sh`
- **Config:** pydantic-settings via `src/config.py` — always import `settings` from there

---

## Non-Obvious Things

- `STORAGE_MODE=local` for dev/CI, `STORAGE_MODE=s3` for production branch
- Integration tests use `@pytest.mark.skip` — never run in CI, run manually only
- The Gold Delta table is the feature store — all training reads from there, not raw files
- MLflow model registry name is always `"misinformation-roberta-v1"`
- Never call `SparkSession.builder` inline — always use `get_spark()` from `src/spark_session.py`
- Never call `os.getenv()` anywhere — all config goes through `src/config.py`

---

## Steering Files

- `gotchas.md` — mistakes to avoid in this codebase
- `architecture.md` — data flow and module responsibilities
- `tech-stack.md` — pinned versions and approved libraries
