"""
pipeline.py — Prefect flow wiring the full lakehouse pipeline.

Flow DAG:
  ingest_static → ingest_reddit
        ↓
  bronze_to_silver
        ↓
  silver_to_gold
        ↓
  train (on schedule or manual trigger)
        ↓
  register model → promote to Production if metrics pass threshold

TODO: Implement in Step 6 (Orchestration Layer)
"""

from __future__ import annotations

from prefect import flow


@flow(name="misinformation-lakehouse-pipeline")
def run_pipeline() -> None:
    raise NotImplementedError("Implement in Step 6")
