# Running on Databricks

This project was built on open-source equivalents of the Databricks platform.
The mapping is direct:

| This Project | Databricks Equivalent |
|---|---|
| delta-spark local/S3 | Delta Lake on DBFS / Unity Catalog |
| MLflow local server | Databricks Managed MLflow |
| Prefect @flow | Databricks Workflows / Delta Live Tables |
| FastAPI + Docker | Databricks Model Serving |
| GitHub Actions | Databricks Asset Bundles CI/CD |
| PySpark local | Databricks Runtime (DBR) |

---

## Migration Notes

### Delta Lake (`src/processing/`)
`get_spark()` calls `configure_spark_with_delta_pip()` to download JARs from Maven — not needed on Databricks, which bundles them in DBR. Remove that call and drop `delta-spark` from `requirements.txt`. Bronze/Silver/Gold tables map directly to Unity Catalog schemas (e.g. `catalog.misinformation.bronze`).

### MLflow (`src/training/train.py`, `src/serving/model_loader.py`)
No code changes required. Replace `MLFLOW_TRACKING_URI=http://localhost:5000` with the workspace URL. The `"misinformation-roberta-v1"` registry name, alias-based promotion (`production`/`staging`), and `models:/name@production` load URIs all work identically against Databricks Managed MLflow.

### Orchestration (`src/orchestration/pipeline.py`)
The Prefect `@flow` / `@task` graph (ingest → bronze_to_silver → silver_to_gold) maps to a Databricks Workflow with three notebook or Python task nodes and the same dependency edges. The daily 02:00 UTC schedule in `schedules.py` maps to a cron trigger on the Workflow.

### Model Serving (`src/serving/app.py`)
The FastAPI app (endpoints: `/health`, `/predict`, `/explain`) and `ModelLoader` can be deployed as-is to Databricks Model Serving by pointing the serving endpoint at the registered `"misinformation-roberta-v1"` model. The Groq `/explain` endpoint remains an external call and is unaffected.

### CI/CD (`.github/workflows/ci.yml`)
The four-stage GitHub Actions pipeline (lint → unit tests → Docker build → model validation) maps to a Databricks Asset Bundle with the same stages. The `validate_model.py` gate runs as a Databricks Workflow job triggered on merge to `main`.

### Storage
`STORAGE_MODE=local` (dev) → DBFS paths (`dbfs:/misinformation/…`).  
`STORAGE_MODE=s3` (production branch) → S3 or Unity Catalog external locations with identical row counts: Bronze 36,032 → Silver 34,534 → Gold 34,534.
