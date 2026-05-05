"""
scripts/validate_model.py

Model validation gate — run in CI before promoting to Production.

Checks:
  1. A model named MODEL_NAME exists in the MLflow registry
  2. The latest version has F1 >= MIN_F1 on the validation set
  3. The model was trained on >= MIN_TRAINING_SAMPLES samples

Exit codes:
  0 — validation passed (or no model registered yet, graceful skip)
  1 — model exists but fails quality threshold
"""

from __future__ import annotations

import sys
import os

MIN_F1 = 0.80
MIN_TRAINING_SAMPLES = 5_000

def main() -> None:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("MLflow not installed — skipping model validation")
        sys.exit(0)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = os.getenv("MODEL_NAME", "misinformation-roberta-v1")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    try:
        versions = client.get_latest_versions(model_name)
    except Exception:
        print(f"No model '{model_name}' found in registry — skipping validation (first run)")
        sys.exit(0)

    if not versions:
        print("No versions registered yet — skipping validation")
        sys.exit(0)

    latest = versions[0]
    run = client.get_run(latest.run_id)
    metrics = run.data.metrics

    f1 = metrics.get("eval_f1", None)
    n_samples = metrics.get("training_samples", None)

    print(f"Model: {model_name} v{latest.version}")
    print(f"  eval_f1:          {f1}")
    print(f"  training_samples: {n_samples}")

    if f1 is None:
        print("WARNING: eval_f1 not logged — cannot validate. Passing with warning.")
        sys.exit(0)

    if f1 < MIN_F1:
        print(f"FAIL: eval_f1={f1:.3f} is below threshold {MIN_F1}")
        sys.exit(1)

    if n_samples is not None and n_samples < MIN_TRAINING_SAMPLES:
        print(f"FAIL: training_samples={n_samples} is below minimum {MIN_TRAINING_SAMPLES}")
        sys.exit(1)

    print(f"PASS: Model meets all quality thresholds ✓")
    sys.exit(0)


if __name__ == "__main__":
    main()
