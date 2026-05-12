"""
bronze_to_silver.py — Clean and deduplicate Bronze → Silver Delta table.

Transformations (in order):
  1. Drop exact duplicates on (text, source)
  2. Drop rows where text is null or empty after stripping
  3. Normalize labels → label_binary (0=credible, 1=misinfo, -1=unknown)
  4. text_clean: strip HTML, normalize whitespace, lowercase
  5. word_count: words in text_clean
  6. char_count: characters in text_clean
  7. processed_at: current timestamp

Writes to Silver Delta table via merge on (id, source), partitioned by source.
"""

from __future__ import annotations

from loguru import logger

_CREDIBLE_LABELS = {"true", "mostly-true", "half-true", "real"}
_MISINFO_LABELS = {"false", "pants-fire", "barely-true", "fake"}


def normalize_label(label: str | None) -> int:
    """Map raw dataset label to binary integer (0, 1, or -1)."""
    if label is None:
        return -1
    label_lower = label.lower().strip()
    if label_lower in _CREDIBLE_LABELS:
        return 0
    if label_lower in _MISINFO_LABELS:
        return 1
    return -1


def run() -> None:
    # Lazy imports keep unit tests free of PySpark
    from delta.tables import DeltaTable
    from pyspark.sql import functions as F  # noqa: N812
    from pyspark.sql.types import IntegerType

    from src.config import settings
    from src.spark_session import get_spark

    spark = get_spark("BronzeToSilver")
    bronze_path = settings.delta_path("bronze")
    silver_path = settings.delta_path("silver")

    bronze_df = spark.read.format("delta").load(bronze_path)
    count_before = bronze_df.count()
    logger.info(f"Bronze rows loaded: {count_before}")

    # 1. Drop exact duplicates on (text, source)
    deduped_df = bronze_df.dropDuplicates(["text", "source"])
    count_after = deduped_df.count()
    logger.info(f"After deduplication: {count_after} rows (dropped {count_before - count_after})")

    # 2. Drop null / empty text
    deduped_df = deduped_df.filter(F.col("text").isNotNull() & (F.trim(F.col("text")) != ""))

    # 3-7. Apply all column transformations
    normalize_label_udf = F.udf(normalize_label, IntegerType())

    silver_df = (
        deduped_df.withColumn("label_binary", normalize_label_udf(F.col("label")))
        .withColumn(
            "text_clean",
            F.lower(
                F.trim(
                    F.regexp_replace(
                        F.regexp_replace(F.col("text"), r"<[^>]+>", ""),
                        r"\s+",
                        " ",
                    )
                )
            ),
        )
        .withColumn("word_count", F.size(F.split(F.col("text_clean"), r"\s+")))
        .withColumn("char_count", F.length(F.col("text_clean")))
        .withColumn("processed_at", F.current_timestamp())
        .select(
            "id",
            "text",
            "text_clean",
            "label",
            "label_binary",
            "source",
            "word_count",
            "char_count",
            "ingested_at",
            "processed_at",
            "raw_meta",
        )
    )

    # Merge into Silver Delta table on (id, source)
    if DeltaTable.isDeltaTable(spark, silver_path):
        (
            DeltaTable.forPath(spark, silver_path)
            .alias("target")
            .merge(
                silver_df.alias("source"),
                "target.id = source.id AND target.source = source.source",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        silver_df.write.format("delta").partitionBy("source").save(silver_path)

    logger.info(f"Silver table written to {silver_path}")


if __name__ == "__main__":
    run()
