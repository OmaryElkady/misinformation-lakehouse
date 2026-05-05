"""
ingest_static.py — Load public misinformation datasets → Bronze Delta table.

Supported datasets:
  - LIAR (Wang 2017): 12,836 labeled political statements
  - FakeNewsNet (Shu et al.): PolitiFact + GossipCop articles

Bronze schema (raw, no transformations):
  id            STRING     source-assigned identifier
  text          STRING     raw statement or article body
  label         STRING     original label (e.g. "pants-fire", "true")
  source        STRING     dataset origin ("liar" | "fakenewsnet")
  ingested_at   TIMESTAMP  wall-clock time of ingestion
  raw_meta      STRING     JSON blob of all original columns

TODO: Implement in Step 2 (Ingestion Layer)
"""

from __future__ import annotations


def run() -> None:
    """Entry point — called by Prefect flow and CLI."""
    raise NotImplementedError("Implement in Step 2")
