# Gotchas

> Things Claude gets wrong in THIS codebase specifically.
> Rule: if you corrected the same mistake twice, it goes here.
> Keep entries short and direct. Delete ones that stop being relevant.

---

## Never Do This

- Do NOT use `os.getenv()` directly — always import `settings` from `src/config.py`
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
