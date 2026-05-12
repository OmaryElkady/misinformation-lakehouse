# Gotchas

> Things Claude gets wrong in THIS codebase specifically.
> Rule: if you corrected the same mistake twice, it goes here.
> Keep entries short and direct. Delete ones that stop being relevant.

---

## Never Do This

- Do NOT use `os.getenv()` directly — always import `settings` from `src/config.py`
- Do NOT use `@app.on_event("startup")` in FastAPI — use the `lifespan` context manager; `on_event` is deprecated in FastAPI 0.111 and removed in later versions
- Do NOT import `mlflow` at the top of any file in `src/serving/` — import it lazily inside the function that needs it; `setuptools >= 80` removes `pkg_resources` and breaks the mlflow module-level import, which prevents test collection
- Do NOT add `from datasets import load_dataset` or `from groq import Groq` as bare module-level imports — wrap them in `try/except ImportError` so the module loads in CI's minimal env; the fallback `= None` keeps the name available for `patch()` in tests
- Do NOT call `SparkSession.builder` inline — always use `get_spark()` from `src/spark_session.py`
- Do NOT write integration tests without `@pytest.mark.skip` — they must never run in CI
- Do NOT add new environment variables without also adding them to `.env.example` AND `src/config.py`
- Do NOT use `pandas` for large data transformations — use PySpark
- Do NOT write Delta table paths as string literals — always use `settings.delta_path("bronze")` etc.

---

## Always Do This

- All new source files go under `src/` in the appropriate subfolder
- All unit tests go in `tests/unit/` and must be decorated with `@pytest.mark.unit`
- All integration tests go in `tests/integration/` with `@pytest.mark.integration` AND `@pytest.mark.skip`
- Every new env var must be added to `.env.example` AND as a field in the `Settings` class in `src/config.py`
- Use `loguru` for all logging — never `print()` or `logging.basicConfig()`

---

## Patterns That Look Right But Aren't

- `STORAGE_MODE` controls whether Delta paths resolve to local or S3 — never hardcode `s3a://` or `./data/delta` directly
- The `settings` object in `src/config.py` is a singleton — import it, don't instantiate `Settings()` again
- Unit tests must not import PySpark — if a test needs Spark it belongs in `tests/integration/`
- `docker compose up` starts MLflow on port 5000, Prefect on 4200, and the API on 8000 — don't change these ports

---

## `patch()` Requires the Module to Be Importable

`unittest.mock.patch("pkg.attr")` calls `importlib.import_module("pkg")` internally to find the
attribute to replace. If the package isn't installed, it raises `ModuleNotFoundError` even when
the function under test only imports that package lazily.

**Fix:** inject the module into `sys.modules` before patching:

```python
# WRONG — fails if mlflow not installed, even though register_model() imports it lazily
with patch("mlflow.set_tracking_uri"), patch("mlflow.register_model", ...):
    register_model(run_id)

# CORRECT — pre-populate sys.modules; lazy imports inside the function see the mock
mock_mlflow = MagicMock()
mock_mlflow.register_model.return_value = mock_result
mock_mlflow.MlflowClient.return_value = mock_client
with patch.dict(sys.modules, {"mlflow": mock_mlflow, "mlflow.tracking": MagicMock()}):
    register_model(run_id)
```

This applies to any lazily-imported dep: `mlflow`, `datasets`, `groq`, `pyspark`, etc.

---

## Serving Unit Test Mocking — Two Different Patterns

Endpoint tests and `ModelLoader` unit tests require different mock strategies.

**Pattern 1 — endpoint tests**: patch the module-level singleton in `app.py`

```python
with patch("src.serving.app.model_loader") as mock_loader:
    mock_loader.loaded = True
    mock_loader.predict.return_value = ("misinformation", 0.92)
    ...
```

This also silences the lifespan's `model_loader.load()` call (the mock's `load()` is a no-op).

**Pattern 2 — ModelLoader.load() unit tests**: mlflow is imported lazily *inside* `load()`,
so there is no `src.serving.model_loader.mlflow` name to patch. Inject mocks via `sys.modules`
*before* the import runs:

```python
mock_tracking = MagicMock()
mock_tracking.MlflowClient.return_value = mock_client  # or .side_effect = Exception(...)

with patch.dict(sys.modules, {"mlflow": MagicMock(), "mlflow.tracking": mock_tracking}):
    loader.load()
```

`from mlflow.tracking import MlflowClient` resolves via `sys.modules`, so the mock intercepts
it without needing the real mlflow to be importable at all.

---

## WSL Venv — Always Use `python3.12` Explicitly

On this machine `/usr/bin/python3` in WSL resolves to Python 3.14.4. The venv symlink chain
`python → python3 → /usr/bin/python3` inherits this, so `source .venv/bin/activate` silently
activates the venv but `python` still runs 3.14 — not 3.12. Git Bash activation fails entirely
(Linux paths, Windows fallback). Always verify with `python --version` after activating.

```bash
# WRONG — creates venv with wrong Python; python3 → 3.14.4 on this machine
python3 -m venv .venv

# CORRECT — pin explicitly
python3.12 -m venv .venv
source .venv/bin/activate   # in WSL only, not Git Bash
python --version             # must show 3.12.x
```

If the venv is already broken: `rm -rf .venv && python3.12 -m venv .venv && pip install -r requirements.txt`

---

## Delta Lake — `configure_spark_with_delta_pip` Is Required

`delta-spark` installed via pip provides Python bindings only — the Scala/Java JARs are not
bundled. Without calling `configure_spark_with_delta_pip(builder)`, Spark raises:

```
ClassNotFoundException: io.delta.sql.DeltaSparkSessionExtension
TypeError: 'JavaPackage' object is not callable
```

`get_spark()` in `src/spark_session.py` handles this. Never bypass it by building a raw
`SparkSession.builder` inline — the Delta extension won't load.

---

## HuggingFace Datasets — Always Pass `trust_remote_code=True`

Both `"liar"` and `"rickstello/FakeNewsNet"` require `trust_remote_code=True`. Without it,
`"liar"` emits a `FutureWarning` (and will hard-fail in a future `datasets` release);
`"rickstello/FakeNewsNet"` may raise immediately. Always:

```python
load_dataset("liar", trust_remote_code=True)
load_dataset("rickstello/FakeNewsNet", trust_remote_code=True)
```

The old `mrjunos/fakenewsnet` dataset no longer exists on HuggingFace Hub — it will raise
`DatasetNotFoundError`. Use `rickstello/FakeNewsNet` instead.

---

## Runnable Modules — Every `run()` Needs a `__main__` Block

Any module with a `run()` entry point that should be invocable with `python -m src.foo.bar`
must end with:

```python
if __name__ == "__main__":
    run()
```

Without it `python -m` imports the module and exits silently — no error, no output, nothing runs.

---

## Prefect 3.x API — Removed and Changed APIs

**Never use Prefect 2.x deployment API.** It does not exist in Prefect 3.x.
**Never use `flow.deploy()` for local runner deployments.** It requires an image or remote storage.

```python
# WRONG — Prefect 2.x only; raises ImportError on Prefect 3.x
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
Deployment.build_from_flow(flow=run_pipeline, schedule=CronSchedule(...)).apply()

# WRONG — Prefect 3.x flow.deploy() for local code; raises ValueError:
#   "Either an image or remote storage location must be provided"
await run_pipeline.deploy(name="...", work_pool_name="...", cron="...")

# CORRECT — Prefect 3.x local runner deployment (to_deployment is also a coroutine)
deployment = await run_pipeline.to_deployment(name="...", cron="...", parameters={...})
await deployment.apply()
```

Both `to_deployment()` and `apply()` are coroutines — await both, and wrap the whole sequence
in `asyncio.run()` when calling from sync code. Use `AsyncMock` (not `MagicMock`) in tests for
both of them.

---

## Prefect 3.x Unit Test Mocking — Task vs Flow Split

In Prefect 3.x, `@task` returns a `Task` object and `@flow` returns a `Flow` object.
The mocking strategy differs between them:

- **Tasks called via `.submit()`** (ingest_static, ingest_bluesky): mock as `MagicMock`; the
  test manipulates `.submit.return_value.result.side_effect` to simulate failure.
- **Tasks called directly** (bronze_to_silver, silver_to_gold): mock as `MagicMock`; simulate
  failure with `.side_effect = RuntimeError(...)` directly on the mock, NOT on `.submit(...)`.
- **Flows** (run_pipeline): wrap in a `_MockFlow` class with `__call__` running the real function
  body and `.deploy` as an `AsyncMock`.

```python
# Task decorator mock — each @task(...) call gets its own independent MagicMock
def _make_task_decorator(*args, **kwargs):
    def decorator(fn):
        return MagicMock(name=f"task_{fn.__name__}")
    return decorator

# Flow decorator mock — callable wrapping real body + AsyncMock .deploy
def _make_flow_decorator(*args, **kwargs):
    def decorator(fn):
        class _MockFlow:
            def __call__(self, *a, **kw): return fn(*a, **kw)
        obj = _MockFlow()
        obj.deploy = AsyncMock()
        return obj
    return decorator
```

---

## Docker Health Checks — No curl in Slim Images

`python:3.11-slim` and `prefecthq/prefect` images do not include `curl`. Health checks using
`curl -f http://localhost:PORT/health` will always fail with `executable file not found`.

```yaml
# WRONG
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5000/health"]

# CORRECT
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
```

---

## MLflow Docker Binding — MLFLOW_HOST Is Required

MLflow ignores `--host 0.0.0.0` in some versions and binds to `127.0.0.1` (loopback) inside
the container. The port is forwarded to the host but the app isn't listening on it, so all
connections are refused. Fix: set `MLFLOW_HOST: "0.0.0.0"` as an environment variable in
`docker-compose.yml` alongside the `--host 0.0.0.0` flag in the command.

---

## PySpark Unit Test Mocking — Two Known Traps

**Trap 1 — Subclassing MagicMock to override comparison operators does NOT work.**

`MagicMock.__init__` calls `_mock_set_magics()` which rewrites every class-level dunder method
(including `__gt__`, `__lt__`, etc.) with a `MagicProxy`. This silently discards your override.

```python
# WRONG — __gt__ will be replaced by MagicProxy at runtime
class _ColumnMock(MagicMock):
    def __gt__(self, other): return MagicMock()

# CORRECT — plain Python class, no metaclass interference
class _FakeColumn:
    def __gt__(self, other): return _FakeColumn()
    def __lt__(self, other): return _FakeColumn()
    def __ge__(self, other): return _FakeColumn()
    def __le__(self, other): return _FakeColumn()
    def __and__(self, other): return _FakeColumn()
    def __or__(self, other): return _FakeColumn()
    def __sub__(self, other): return _FakeColumn()
    def __truediv__(self, other): return _FakeColumn()
    def __call__(self, *a, **k): return _FakeColumn()
    def __getattr__(self, name): return lambda *a, **k: _FakeColumn()
```

Use `_FakeColumn` instances as the `return_value` for `mock_F.col`, `mock_F.length`, etc.

**Trap 2 — `sys.modules` patching alone does not wire up `from pyspark.sql import functions as F`.**

Python resolves `from pyspark.sql import functions` via `getattr(pyspark_sql_module, "functions")`,
not by looking up `sys.modules["pyspark.sql.functions"]` directly. Since `pyspark.sql` in
sys.modules is a `MagicMock`, `getattr(mock, "functions")` returns an auto-generated attribute —
NOT your configured `mock_F`.

```python
# WRONG — F inside run() will be an auto-generated MagicMock, not mock_F
modules = {"pyspark.sql.functions": mock_F, ...}

# CORRECT — also set the attribute on the pyspark.sql mock object
pyspark_sql_mock = MagicMock()
pyspark_sql_mock.functions = mock_F
modules = {
    "pyspark.sql": pyspark_sql_mock,
    "pyspark.sql.functions": mock_F,
    ...
}
```
