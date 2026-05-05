"""
ingest_reddit.py — Stream Reddit posts → Bronze Delta table (live ingestion).

Subreddits monitored: r/news, r/politics, r/worldnews, r/conspiracy
Each post appended to Bronze with the same schema as ingest_static.py.

TODO: Implement in Step 2 (Ingestion Layer)
"""

from __future__ import annotations


def run(subreddits: list[str] | None = None, limit: int = 100) -> None:
    """Entry point — called by Prefect flow and CLI."""
    raise NotImplementedError("Implement in Step 2")
