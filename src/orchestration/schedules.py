"""
schedules.py — Prefect deployment schedules for the lakehouse pipeline.

Two deployments:
  daily-full-pipeline    — runs every day at 02:00 UTC (full ingest + process)
  manual-processing-only — no schedule; triggered manually to reprocess Bronze data
"""

from __future__ import annotations

import asyncio

from loguru import logger


async def _deploy_all() -> None:
    from src.orchestration.pipeline import run_pipeline

    logger.info("Registering daily-full-pipeline deployment")
    await run_pipeline.deploy(
        name="daily-full-pipeline",
        work_pool_name="default-agent-pool",
        cron="0 2 * * *",
        parameters={
            "run_ingestion": True,
            "run_bluesky": True,
            "run_processing": True,
        },
    )
    logger.info("daily-full-pipeline registered")

    logger.info("Registering manual-processing-only deployment")
    await run_pipeline.deploy(
        name="manual-processing-only",
        work_pool_name="default-agent-pool",
        parameters={
            "run_ingestion": False,
            "run_bluesky": False,
            "run_processing": True,
        },
    )
    logger.info("manual-processing-only registered")


def deploy() -> None:
    """Register both pipeline deployments with the Prefect server."""
    asyncio.run(_deploy_all())
