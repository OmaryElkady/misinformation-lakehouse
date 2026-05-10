"""
pipeline.py — Prefect flow wiring the full lakehouse pipeline.

Flow DAG:
  ingest_static ┐  (submitted concurrently; bluesky failure is allow_failure)
  ingest_bluesky┘
        ↓
  bronze_to_silver
        ↓
  silver_to_gold
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger
from prefect import flow, task
from prefect.futures import allow_failure


@task(
    name="ingest-static",
    description="Load LIAR + FakeNewsNet datasets into Bronze Delta table",
    retries=2,
    retry_delay_seconds=30,
)
def ingest_static_task() -> None:
    logger.info("ingest_static_task starting")
    from src.ingestion.ingest_static import run

    run()
    logger.info("ingest_static_task complete")


@task(
    name="ingest-bluesky",
    description="Fetch Bluesky posts into Bronze Delta table",
    retries=2,
    retry_delay_seconds=30,
)
def ingest_bluesky_task(limit: int = 100) -> None:
    logger.info(f"ingest_bluesky_task starting | limit={limit}")
    from src.ingestion.ingest_bluesky import run

    run(limit=limit)
    logger.info("ingest_bluesky_task complete")


@task(
    name="bronze-to-silver",
    description="Clean and deduplicate Bronze → Silver Delta table",
    retries=2,
    retry_delay_seconds=30,
)
def bronze_to_silver_task() -> None:
    logger.info("bronze_to_silver_task starting")
    from src.processing.bronze_to_silver import run

    run()
    logger.info("bronze_to_silver_task complete")


@task(
    name="silver-to-gold",
    description="Feature engineering Silver → Gold Delta table",
    retries=2,
    retry_delay_seconds=30,
)
def silver_to_gold_task() -> None:
    logger.info("silver_to_gold_task starting")
    from src.processing.silver_to_gold import run

    run()
    logger.info("silver_to_gold_task complete")


@flow(name="misinformation-lakehouse-pipeline")
def run_pipeline(
    run_ingestion: bool = True,
    run_bluesky: bool = True,
    run_processing: bool = True,
    limit_bluesky: int = 100,
) -> dict[str, Any]:
    """Orchestrate the full lakehouse pipeline: ingest → process."""
    start = time.perf_counter()
    tasks_completed: list[str] = []
    tasks_failed: list[str] = []

    logger.info(
        f"Pipeline starting | run_ingestion={run_ingestion} "
        f"run_bluesky={run_bluesky} run_processing={run_processing}"
    )

    # ── Ingestion (concurrent; bluesky failure must not block static) ───────────
    static_future = None
    bluesky_future = None
    upstream: list = []

    if run_ingestion:
        logger.info("Submitting ingest_static")
        static_future = ingest_static_task.submit()
        upstream.append(allow_failure(static_future))

        if run_bluesky:
            logger.info("Submitting ingest_bluesky")
            bluesky_future = ingest_bluesky_task.submit(limit=limit_bluesky)
            upstream.append(allow_failure(bluesky_future))

    # ── Processing (sequential; b2s waits for ingestion, s2g waits for b2s) ────
    b2s_future = None
    s2g_future = None

    if run_processing:
        logger.info("Submitting bronze_to_silver")
        b2s_future = (
            bronze_to_silver_task.submit(wait_for=upstream)
            if upstream
            else bronze_to_silver_task.submit()
        )
        logger.info("Submitting silver_to_gold")
        s2g_future = silver_to_gold_task.submit(wait_for=[b2s_future])

    # ── Collect results ─────────────────────────────────────────────────────────
    if static_future is not None:
        try:
            static_future.result()
            tasks_completed.append("ingest_static")
            logger.info("ingest_static completed")
        except Exception as exc:
            tasks_failed.append("ingest_static")
            logger.error(f"ingest_static failed: {exc}")

    if bluesky_future is not None:
        try:
            bluesky_future.result()
            tasks_completed.append("ingest_bluesky")
            logger.info("ingest_bluesky completed")
        except Exception as exc:
            tasks_failed.append("ingest_bluesky")
            logger.warning(f"ingest_bluesky failed (non-critical): {exc}")

    if b2s_future is not None:
        try:
            b2s_future.result()
            tasks_completed.append("bronze_to_silver")
            logger.info("bronze_to_silver completed")
        except Exception as exc:
            tasks_failed.append("bronze_to_silver")
            logger.error(f"bronze_to_silver failed: {exc}")

    if s2g_future is not None:
        try:
            s2g_future.result()
            tasks_completed.append("silver_to_gold")
            logger.info("silver_to_gold completed")
        except Exception as exc:
            tasks_failed.append("silver_to_gold")
            logger.error(f"silver_to_gold failed: {exc}")

    duration = time.perf_counter() - start

    # bluesky failure alone → partial; any other failure → failed
    critical_failures = [t for t in tasks_failed if t != "ingest_bluesky"]
    if not tasks_failed:
        status = "success"
    elif not critical_failures:
        status = "partial"
    else:
        status = "failed"

    logger.info(f"Pipeline complete | status={status} | duration={duration:.2f}s")

    return {
        "status": status,
        "duration_seconds": duration,
        "tasks_completed": tasks_completed,
        "tasks_failed": tasks_failed,
    }
