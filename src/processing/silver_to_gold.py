"""
silver_to_gold.py — Feature engineering Silver → Gold Delta table.

Transformations (in order):
  1. Filter out unknown labels (label_binary == -1)
  2. sentiment_score: TextBlob polarity via UDF
  3. exclamation_ratio: "!" count / char_count
  4. caps_ratio: uppercase letter count / original text length
  5. avg_word_length: char_count / word_count (zero-safe)
  6. source_credibility: static score per source
  7. split: deterministic train/val/test via hash on id (80/10/10)
  8. featurized_at: current timestamp

Writes to Gold Delta table via merge on (id, source),
partitioned by (source, split).
"""

from __future__ import annotations

from loguru import logger

_SOURCE_CREDIBILITY: dict[str, float] = {
    "liar": 0.7,
    "fakenewsnet": 0.6,
    "bluesky": 0.3,
}


def get_source_credibility(source: str) -> float:
    """Return a static credibility score for a known source, 0.5 for unknown."""
    return _SOURCE_CREDIBILITY.get(source, 0.5)


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide numerator by denominator, returning 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def run() -> None:
    # Lazy imports keep unit tests free of PySpark / TextBlob
    from delta.tables import DeltaTable
    from pyspark.sql import functions as F  # noqa: N812
    from pyspark.sql.types import FloatType
    from textblob import TextBlob

    from src.config import settings
    from src.spark_session import get_spark

    spark = get_spark("SilverToGold")
    silver_path = settings.delta_path("silver")
    gold_path = settings.delta_path("gold")

    # 1. Load Silver and filter out unknown labels
    silver_df = spark.read.format("delta").load(silver_path)
    filtered_df = silver_df.filter(F.col("label_binary") != -1)

    # UDFs
    @F.udf(returnType=FloatType())
    def sentiment_udf(text: str | None) -> float:
        if text is None:
            return 0.0
        return float(TextBlob(text).sentiment.polarity)

    @F.udf(returnType=FloatType())
    def source_credibility_udf(source: str | None) -> float:
        return get_source_credibility(source or "")

    # 2-8. Feature engineering
    gold_df = (
        filtered_df.withColumn("sentiment_score", sentiment_udf(F.col("text_clean")))
        .withColumn(
            "exclamation_ratio",
            F.when(
                F.col("char_count") > 0,
                (
                    F.col("char_count") - F.length(F.regexp_replace(F.col("text_clean"), "!", ""))
                ).cast(FloatType())
                / F.col("char_count").cast(FloatType()),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "caps_ratio",
            F.when(
                F.length(F.col("text")) > 0,
                F.length(F.regexp_replace(F.col("text"), "[^A-Z]", "")).cast(FloatType())
                / F.length(F.col("text")).cast(FloatType()),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "avg_word_length",
            F.when(
                F.col("word_count") > 0,
                F.col("char_count").cast(FloatType()) / F.col("word_count").cast(FloatType()),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("source_credibility", source_credibility_udf(F.col("source")))
        .withColumn("_hash_mod", F.pmod(F.hash(F.col("id")), F.lit(10)))
        .withColumn(
            "split",
            F.when(F.col("_hash_mod") <= 7, "train")
            .when(F.col("_hash_mod") == 8, "val")
            .otherwise("test"),
        )
        .drop("_hash_mod")
        .withColumn("featurized_at", F.current_timestamp())
        .select(
            "id",
            "text_clean",
            "label_binary",
            "source",
            "word_count",
            "char_count",
            "sentiment_score",
            "exclamation_ratio",
            "caps_ratio",
            "avg_word_length",
            "source_credibility",
            "split",
            "ingested_at",
            "processed_at",
            "featurized_at",
        )
    )

    # Merge into Gold Delta table on (id, source)
    if DeltaTable.isDeltaTable(spark, gold_path):
        (
            DeltaTable.forPath(spark, gold_path)
            .alias("target")
            .merge(
                gold_df.alias("source"),
                "target.id = source.id AND target.source = source.source",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        gold_df.write.format("delta").partitionBy("source", "split").save(gold_path)

    # Log class distribution
    dist = {
        row["label_binary"]: row["count"]
        for row in gold_df.groupBy("label_binary").count().collect()
    }
    logger.info(f"Gold table written to {gold_path} | class distribution: {dist}")
