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
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

# datasets is a heavy HuggingFace library not installed in CI's minimal env.
# The name must exist at module level so tests can patch it.
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # type: ignore[assignment]

from src.config import settings

# ── Bronze schema constants ────────────────────────────────────────────────────
BRONZE_COLUMNS = ["id", "text", "label", "source", "ingested_at", "raw_meta"]
BRONZE_MERGE_KEYS = ["id", "source"]


def _make_record(
    row_id: str,
    text: str,
    label: str,
    source: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Build a Bronze-schema dict (without ingested_at — added by Spark)."""
    return {
        "id": str(row_id),
        "text": text or "",
        "label": str(label),
        "source": source,
        "raw_meta": json.dumps(meta, default=str),
    }


def _load_liar() -> list[dict[str, Any]]:
    logger.info("Loading LIAR dataset from HuggingFace")
    ds = load_dataset("liar", trust_remote_code=True)
    label_names: list[str] = ds["train"].features["label"].names
    records: list[dict[str, Any]] = []
    for split_name, split in ds.items():
        for row in split:
            label_str = label_names[row["label"]]
            meta = {k: v for k, v in row.items() if k not in ("id", "statement", "label")}
            meta["split"] = split_name
            records.append(
                _make_record(
                    row_id=row["id"],
                    text=row["statement"],
                    label=label_str,
                    source="liar",
                    meta=meta,
                )
            )
    logger.info(f"LIAR: {len(records)} records loaded")
    return records


def _get_col(row: dict[str, Any], *candidates: str, default: str = "") -> str:
    for col in candidates:
        if col in row and row[col] is not None:
            return str(row[col])
    return default


def _load_fakenewsnet() -> list[dict[str, Any]]:
    # rickstello/FakeNewsNet: cols = title, news_url, source_domain, tweet_num, real (0/1 int)
    logger.info("Loading FakeNewsNet dataset from HuggingFace")
    ds = load_dataset("rickstello/FakeNewsNet", trust_remote_code=True)
    records: list[dict[str, Any]] = []
    for split_name, split in ds.items():
        for idx, row in enumerate(split):
            label = "real" if row.get("real") == 1 else "fake"
            meta = {
                "news_url": row.get("news_url", ""),
                "source_domain": row.get("source_domain", ""),
                "tweet_num": row.get("tweet_num", 0),
                "split": split_name,
            }
            records.append(
                _make_record(
                    row_id=f"{split_name}_{idx}",
                    text=_get_col(row, "title"),
                    label=label,
                    source="fakenewsnet",
                    meta=meta,
                )
            )
    logger.info(f"FakeNewsNet: {len(records)} records loaded")
    return records


def _merge_to_bronze(records: list[dict[str, Any]], path: str) -> None:
    """Write records to Bronze Delta table using merge to avoid duplicates on re-run."""
    # Lazy imports keep unit tests free of PySpark
    import pyspark.sql.functions as F  # noqa: N812
    from delta.tables import DeltaTable
    from pyspark.sql.types import StringType, StructField, StructType

    from src.spark_session import get_spark

    spark = get_spark("StaticIngestion")

    record_schema = StructType(
        [
            StructField("id", StringType(), nullable=False),
            StructField("text", StringType(), nullable=True),
            StructField("label", StringType(), nullable=True),
            StructField("source", StringType(), nullable=False),
            StructField("raw_meta", StringType(), nullable=True),
        ]
    )

    df = spark.createDataFrame(records, schema=record_schema).withColumn(
        "ingested_at", F.current_timestamp()
    )

    if DeltaTable.isDeltaTable(spark, path):
        logger.info(f"Merging {len(records)} records into Bronze at {path}")
        delta_table = DeltaTable.forPath(spark, path)
        (
            delta_table.alias("t")
            .merge(df.alias("s"), "t.id = s.id AND t.source = s.source")
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        logger.info(f"Creating Bronze table at {path} with {len(records)} records")
        df.write.format("delta").save(path)


def run() -> None:
    """Entry point — load all static datasets and write to Bronze Delta table."""
    path = settings.delta_path("bronze")
    logger.info(f"Static ingestion starting | target={path}")

    liar_records = _load_liar()
    fnn_records = _load_fakenewsnet()
    all_records = liar_records + fnn_records

    logger.info(f"Total records to ingest: {len(all_records)}")
    _merge_to_bronze(all_records, path)
    logger.info("Static ingestion complete")


if __name__ == "__main__":
    run()
