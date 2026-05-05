"""
tests/integration/test_pipeline_smoke.py

Smoke tests for the full pipeline.
These are SKIPPED in CI (require Spark + network).
Run locally with: pytest -m integration
"""

import pytest


@pytest.mark.integration
@pytest.mark.skip(reason="Requires local Spark — run manually with: pytest -m integration")
def test_bronze_delta_table_created():
    """After ingestion, Bronze Delta table should exist and be readable."""
    from src.spark_session import get_spark
    from src.config import settings
    import os

    spark = get_spark("SmokeTest")
    bronze_path = settings.bronze_path
    assert os.path.exists(bronze_path), f"Bronze table not found at {bronze_path}"

    df = spark.read.format("delta").load(bronze_path)
    assert df.count() > 0
    assert "text" in df.columns
    assert "label" in df.columns
    spark.stop()


@pytest.mark.integration
@pytest.mark.skip(reason="Requires local Spark — run manually with: pytest -m integration")
def test_gold_table_has_required_columns():
    """Gold table must have all feature columns before training."""
    from src.spark_session import get_spark
    from src.config import settings

    REQUIRED_COLUMNS = [
        "id", "text", "label", "word_count", "char_count",
        "sentiment_score", "source",
    ]
    spark = get_spark("SmokeTest")
    df = spark.read.format("delta").load(settings.gold_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Gold table missing columns: {missing}"
    spark.stop()
