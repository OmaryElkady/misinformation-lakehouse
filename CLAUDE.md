# Misinformation Detection Lakehouse

> End-to-end MLOps platform that ingests Reddit posts and public datasets,
> processes them through a Bronze→Silver→Gold Delta Lake pipeline, fine-tunes
> RoBERTa for misinformation classification, and serves predictions via FastAPI.

---

## Quick Start

```bash
# Install (in WSL2, Python 3.12)
# Must use python3.12 explicitly — the venv's `python` symlink can resolve to the
# Windows system Python (3.14) on /mnt/c paths, causing ModuleNotFoundError at runtime
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start services (MLflow + Prefect + API)
docker compose up -d

# Register Prefect deployments (one-time, after services are healthy)
export PREFECT_API_URL=http://localhost:4200/api
python3.12 -c "from src.orchestration.schedules import deploy; deploy()"

# Start the Prefect worker (keep running in a dedicated terminal)
PREFECT_API_URL=http://localhost:4200/api prefect worker start --pool default-agent-pool

# Test
python3.12 -m pytest tests/unit/ -v
```

---

## Project Structure

```
src/
  ingestion/       # LIAR dataset + FakeNewsNet → Bronze Delta table
  processing/      # Bronze → Silver → Gold PySpark jobs
  training/        # RoBERTa fine-tune + MLflow experiment tracking
  serving/         # FastAPI inference server
  orchestration/
    pipeline.py    # Prefect @flow wiring ingest → process steps
    schedules.py   # deploy() registers daily + manual deployments
  config.py        # ALL env vars loaded here — never use os.getenv() elsewhere
  spark_session.py # SparkSession factory — always use get_spark(), never inline
scripts/
  run_pipeline.py  # Typer CLI (full / process-only / ingest-only / status)
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
- **WSL venv must use `python3.12` explicitly** — `/usr/bin/python3` on this machine resolves to Python 3.14.4; the venv `python` symlink inherits this and silently uses the wrong interpreter. Always create with `python3.12 -m venv .venv`. If broken, `rm -rf .venv` and recreate. Git Bash venv activation also silently falls back to Windows Python — always run pipeline code in WSL.
- **Delta Lake JARs need `configure_spark_with_delta_pip`** — `delta-spark` pip installs Python bindings only; Scala JARs must be fetched from Maven at startup. `get_spark()` in `spark_session.py` calls `configure_spark_with_delta_pip(builder)` before `getOrCreate()`. Without it, Spark raises `ClassNotFoundException: io.delta.sql.DeltaSparkSessionExtension` and all Delta ops fail with `TypeError: 'JavaPackage' object is not callable`. JARs are cached in `~/.ivy2.5.2/jars` after first run.
- **FakeNewsNet dataset is `rickstello/FakeNewsNet`** — `mrjunos/fakenewsnet` no longer exists on HuggingFace Hub. Schema: `title` (text), `news_url`, `source_domain`, `tweet_num`, `real` (int: 0=fake, 1=real), single `train` split, 23,196 records.
- **HuggingFace `trust_remote_code=True`** — required for both `load_dataset("liar", ...)` and `load_dataset("rickstello/FakeNewsNet", ...)`; will be mandatory in the next major `datasets` release.
- **`praw` and `atproto` removed from `requirements.txt`** — both were unused (social media ingestion descoped) and `atproto==0.0.54` caused a hard conflict (`httpx<0.27.0` vs pinned `httpx==0.27.0`).
- **Prefect 3.x only** — Prefect 2.x is incompatible with Python 3.12 (`pydantic.v1` calls `ForwardRef._evaluate()` with a signature that changed in 3.12); use `prefect>=3.0,<4.0`
- Prefect 3.x deployment API: use `await flow.deploy(name=..., work_pool_name=..., cron=..., parameters=...)` — `Deployment.build_from_flow()` and `CronSchedule` from `prefect.schedules` do not exist in Prefect 3
- The work pool `default-agent-pool` must be created once before `deploy()` will succeed: `prefect work-pool create default-agent-pool --type process`
- Docker health checks must not use `curl` — neither `python:3.11-slim` nor `prefecthq/prefect` images include it; use `python -c "import urllib.request; urllib.request.urlopen(...)"` instead
- MLflow requires `MLFLOW_HOST: "0.0.0.0"` in `docker-compose.yml` env — without it the server binds to `127.0.0.1` inside the container and is unreachable from the host even with `--host 0.0.0.0` in the command
- **MLflow client/server versions must match** — the Colab notebook installs `mlflow>=2.15.0`; the Docker server must be the same. A 2.15+ client calling a 2.13.0 server raises `MlflowException: 404` on `/api/2.0/mlflow/logged-models` because that endpoint didn't exist until 2.15. Both `docker-compose.yml` and `requirements.txt` are pinned to `>=2.15.0`
- **MLflow 2.15+ host header validation** — MLflow 2.15+ rejects requests whose `Host` header doesn't match `localhost` (DNS-rebinding protection). When routing Colab through ngrok, always start ngrok with `ngrok http 5000 --host-header="localhost:5000"` so the header is rewritten before hitting MLflow. Without this flag, every Colab API call returns 403
- **MLflow 2.15+ CSRF** — the `/ajax-api/` endpoints (used by the MLflow React UI) return 403 for POST requests from non-local origins including the ngrok URL. Always view the MLflow UI at `http://localhost:5000` directly in a browser; ngrok is for Colab's programmatic API access only. Use an incognito window if you previously visited the ngrok URL to avoid stale origin context
- **MLflow 2.15+ docker-compose** — the server command must include `--extra-allowed-hosts '*'` to pass the host-header allow-list for all local origins
- **`transition_model_version_stage` deprecated since MLflow 2.9** — used in `register_model()` in `src/training/train.py`; model registry stages will be removed in a future major MLflow release. Works now but will eventually need migrating to the aliases API
- **Colab notebook: do not pin torch** — Colab pre-installs a CUDA-compatible torch (e.g. `torch==2.10.0+cu128` for CUDA 12.8). Pinning `torch==2.3.0` downgrades it to a build for CUDA 11.8/12.1, breaking GPU training silently. Remove torch from Cell 1's pip install entirely
- **Colab notebook: use post-numpy-2.0 package versions** — packages released before numpy 2.0 (June 2024) carry a hard `numpy<2` constraint. Colab's pre-installed pandas and other packages are compiled against numpy 2.x; if numpy is downgraded to 1.26.4 you get `ValueError: numpy.dtype size changed, may indicate binary incompatibility`. Use `transformers>=4.44.0`, `datasets>=2.21.0`, `mlflow>=2.15.0`, `accelerate>=0.34.0`, `scikit-learn>=1.6.0` in the notebook
- **Colab notebook: restart runtime after Cell 1** — after the pip install cell runs, you must do Runtime → Restart session before running Cell 2 onward. The old numpy binary stays loaded in memory until restart; skipping this causes the dtype size mismatch error even after a correct install
- **`evaluation_strategy` renamed to `eval_strategy` in transformers ≥4.46** — `TrainingArguments` raises `TypeError: unexpected keyword argument 'evaluation_strategy'` in newer transformers. The notebook uses `eval_strategy='epoch'`

---

## Steering Files

- `gotchas.md` — mistakes to avoid in this codebase (includes PySpark mock traps)
- `architecture.md` — data flow and module responsibilities
- `tech-stack.md` — pinned versions and approved libraries
- `scripts/setup_colab.md` — step-by-step Colab training walkthrough


