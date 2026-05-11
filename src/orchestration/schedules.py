"""
schedules.py — Prefect deployment schedules for the lakehouse pipeline.

Two deployments:
  daily-full-pipeline    — runs every day at 02:00 UTC (full ingest + process)
  manual-processing-only — no schedule; triggered manually to reprocess Bronze data

Prefect 3.x note: flow.deploy() requires an image or remote storage.
For local runner-managed deployments use to_deployment() (awaitable) + apply().
"""

from __future__ import annotations

import asyncio

from loguru import logger


async def _deploy_all() -> None:
    from src.orchestration.pipeline import run_pipeline

    logger.info("Registering daily-full-pipeline deployment")
    daily = await run_pipeline.to_deployment(
        name="daily-full-pipeline",
        cron="0 2 * * *",
        parameters={
            "run_ingestion": True,
            "run_bluesky": True,
            "run_processing": True,
        },
    )
    await daily.apply()
    logger.info("daily-full-pipeline registered")

    logger.info("Registering manual-processing-only deployment")
    manual = await run_pipeline.to_deployment(
        name="manual-processing-only",
        parameters={
            "run_ingestion": False,
            "run_bluesky": False,
            "run_processing": True,
        },
    )
    await manual.apply()
    logger.info("manual-processing-only registered")


def deploy() -> None:
    """Register both pipeline deployments with the Prefect server."""
    asyncio.run(_deploy_all())
