"""
schedules.py — Prefect deployment schedules for the lakehouse pipeline.

Two deployments:
  daily-full-pipeline    — runs every day at 02:00 UTC (full ingest + process)
  manual-processing-only — no schedule; triggered manually to reprocess Bronze data
"""

from __future__ import annotations

from loguru import logger


def deploy() -> None:
    """Register both pipeline deployments with the Prefect server."""
    from prefect.deployments import Deployment
    from prefect.schedules import CronSchedule

    from src.orchestration.pipeline import run_pipeline

    logger.info("Registering daily-full-pipeline deployment")
    daily = Deployment.build_from_flow(
        flow=run_pipeline,
        name="daily-full-pipeline",
        schedule=CronSchedule("0 2 * * *"),
        parameters={
            "run_ingestion": True,
            "run_bluesky": True,
            "run_processing": True,
        },
    )
    daily.apply()
    logger.info("daily-full-pipeline registered")

    logger.info("Registering manual-processing-only deployment")
    manual = Deployment.build_from_flow(
        flow=run_pipeline,
        name="manual-processing-only",
        parameters={
            "run_ingestion": False,
            "run_bluesky": False,
            "run_processing": True,
        },
    )
    manual.apply()
    logger.info("manual-processing-only registered")
