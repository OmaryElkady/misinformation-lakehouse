"""
run_pipeline.py — CLI for running the lakehouse pipeline without a Prefect server.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make src.* importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Misinformation lakehouse pipeline CLI")
console = Console()


@app.command()
def full() -> None:
    """Run the complete pipeline (ingest + process)."""
    console.print("[bold blue]Starting full pipeline...[/bold blue]")
    start = time.perf_counter()
    _run_ingestion()
    _run_processing()
    duration = time.perf_counter() - start
    console.print(f"[bold green]Full pipeline complete in {duration:.2f}s[/bold green]")


@app.command(name="process-only")
def process_only() -> None:
    """Run processing only (bronze→silver→gold). Useful for reprocessing without re-ingesting."""
    console.print("[bold blue]Starting processing only...[/bold blue]")
    _run_processing()
    console.print("[bold green]Processing complete[/bold green]")


@app.command(name="ingest-only")
def ingest_only() -> None:
    """Run ingestion only (static datasets + Bluesky)."""
    console.print("[bold blue]Starting ingestion only...[/bold blue]")
    _run_ingestion()
    console.print("[bold green]Ingestion complete[/bold green]")


@app.command()
def status() -> None:
    """Check whether Bronze/Silver/Gold Delta tables exist and report their row counts."""
    from src.config import settings
    from src.spark_session import get_spark

    spark = get_spark("StatusCheck")

    tbl = Table(title="Delta Lake Table Status")
    tbl.add_column("Layer", style="cyan", no_wrap=True)
    tbl.add_column("Path", style="dim")
    tbl.add_column("Exists", justify="center")
    tbl.add_column("Row Count", justify="right", style="bold")

    for layer in ("bronze", "silver", "gold"):
        path = settings.delta_path(layer)
        try:
            from delta.tables import DeltaTable

            exists = DeltaTable.isDeltaTable(spark, path)
        except Exception:
            exists = False

        if exists:
            try:
                count = str(spark.read.format("delta").load(path).count())
            except Exception as exc:
                count = f"error: {exc}"
        else:
            count = "—"

        tbl.add_row(
            layer.capitalize(),
            path,
            "[green]✓[/green]" if exists else "[red]✗[/red]",
            count,
        )

    console.print(tbl)


# ── helpers ────────────────────────────────────────────────────────────────────


def _run_ingestion() -> None:
    from src.ingestion.ingest_bluesky import run as run_bluesky
    from src.ingestion.ingest_static import run as run_static

    console.print("  [cyan]Running static ingestion...[/cyan]")
    logger.info("CLI: static ingestion starting")
    run_static()
    logger.info("CLI: static ingestion complete")

    console.print("  [cyan]Running Bluesky ingestion...[/cyan]")
    logger.info("CLI: Bluesky ingestion starting")
    try:
        run_bluesky()
    except Exception as exc:
        logger.warning(f"CLI: Bluesky ingestion failed (non-critical): {exc}")
        console.print(f"  [yellow]Bluesky ingestion skipped: {exc}[/yellow]")
    logger.info("CLI: Bluesky ingestion done")


def _run_processing() -> None:
    from src.processing.bronze_to_silver import run as run_b2s
    from src.processing.silver_to_gold import run as run_s2g

    console.print("  [cyan]Running Bronze→Silver...[/cyan]")
    logger.info("CLI: bronze_to_silver starting")
    run_b2s()
    logger.info("CLI: bronze_to_silver complete")

    console.print("  [cyan]Running Silver→Gold...[/cyan]")
    logger.info("CLI: silver_to_gold starting")
    run_s2g()
    logger.info("CLI: silver_to_gold complete")


if __name__ == "__main__":
    app()
