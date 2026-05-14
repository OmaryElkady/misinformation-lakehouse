"""
tests/unit/test_ingestion.py

Unit tests for the ingestion layer.
No Spark, no network — all external calls are mocked.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestBronzeSchema:
    def test_all_required_columns_defined(self):
        from src.ingestion.ingest_static import BRONZE_COLUMNS

        required = {"id", "text", "label", "source", "ingested_at", "raw_meta"}
        assert required.issubset(set(BRONZE_COLUMNS))

    def test_merge_keys_are_id_and_source(self):
        from src.ingestion.ingest_static import BRONZE_MERGE_KEYS

        assert "id" in BRONZE_MERGE_KEYS
        assert "source" in BRONZE_MERGE_KEYS


@pytest.mark.unit
class TestLabelPreservation:
    def test_label_preserved_as_is(self):
        from src.ingestion.ingest_static import _make_record

        for label in ["pants-fire", "true", "half-true", "mostly-true", "fake", "real", "unknown"]:
            record = _make_record("id1", "some text", label, "liar", {})
            assert record["label"] == label, f"Label {label!r} was altered to {record['label']!r}"

    def test_label_not_coerced_to_integer(self):
        from src.ingestion.ingest_static import _make_record

        record = _make_record("x", "text", "pants-fire", "liar", {})
        assert record["label"] != "0"
        assert record["label"] != 0

    def test_label_not_normalized_to_boolean(self):
        from src.ingestion.ingest_static import _make_record

        for label in ["false", "pants-fire", "FAKE"]:
            record = _make_record("x", "text", label, "liar", {})
            assert record["label"] not in (False, "False", 0, "0")


@pytest.mark.unit
class TestRawMetaSerialization:
    def test_raw_meta_is_valid_json(self):
        from src.ingestion.ingest_static import _make_record

        meta = {"speaker": "John Doe", "count": 5, "context": "debate"}
        record = _make_record("id1", "claim text", "true", "liar", meta)
        parsed = json.loads(record["raw_meta"])
        assert parsed["speaker"] == "John Doe"
        assert parsed["count"] == 5
        assert parsed["context"] == "debate"

    def test_raw_meta_non_serializable_values_handled(self):
        from src.ingestion.ingest_static import _make_record

        meta = {"obj": object()}
        record = _make_record("id1", "text", "true", "liar", meta)
        # default=str means non-serializable values become their repr — should not raise
        parsed = json.loads(record["raw_meta"])
        assert "obj" in parsed

    def test_raw_meta_empty_meta_produces_empty_object(self):
        from src.ingestion.ingest_static import _make_record

        record = _make_record("id1", "text", "true", "liar", {})
        assert json.loads(record["raw_meta"]) == {}

    def test_raw_meta_nested_values_serialized(self):
        from src.ingestion.ingest_static import _make_record

        meta = {"counts": {"true": 3, "false": 1}, "tags": ["politics", "health"]}
        record = _make_record("id1", "text", "half-true", "liar", meta)
        parsed = json.loads(record["raw_meta"])
        assert parsed["counts"]["true"] == 3
        assert "politics" in parsed["tags"]


@pytest.mark.unit
class TestRunRaisesOnHFFailure:
    def test_run_raises_when_liar_dataset_fails_to_load(self):
        from src.ingestion.ingest_static import run

        with patch("src.ingestion.ingest_static.load_dataset") as mock_load:
            mock_load.side_effect = RuntimeError("HuggingFace connection failed")
            with pytest.raises(RuntimeError, match="HuggingFace connection failed"):
                run()
        mock_load.assert_called_once_with("liar", trust_remote_code=True)

    def test_run_raises_when_fakenewsnet_dataset_fails_to_load(self):
        import unittest.mock as mock

        from src.ingestion.ingest_static import run

        # Minimal mock of the LIAR dataset structure

        liar_label_feature = mock.MagicMock()
        liar_label_feature.names = [
            "false",
            "half-true",
            "mostly-true",
            "true",
            "barely-true",
            "pants-fire",
        ]

        liar_split = mock.MagicMock()
        liar_split.features = {"label": liar_label_feature}
        liar_split.__iter__ = mock.Mock(return_value=iter([]))

        liar_ds = mock.MagicMock()
        liar_ds.__getitem__ = mock.Mock(return_value=liar_split)
        liar_ds.items.return_value = [("train", liar_split)]

        with patch("src.ingestion.ingest_static.load_dataset") as mock_load:
            mock_load.side_effect = [liar_ds, RuntimeError("FakeNewsNet not available")]
            with pytest.raises(RuntimeError, match="FakeNewsNet not available"):
                run()


# ── Helpers for mocked run() / _merge_to_bronze() tests ───────────────────────


def _make_liar_mock() -> MagicMock:
    label_feature = MagicMock()
    label_feature.names = ["false", "half-true", "mostly-true", "true", "barely-true", "pants-fire"]
    row = {"id": "1", "statement": "Test claim", "label": 0}
    split = MagicMock()
    split.features = {"label": label_feature}
    split.__iter__ = MagicMock(return_value=iter([row]))
    ds = MagicMock()
    ds.__getitem__ = MagicMock(return_value=split)
    ds.items.return_value = [("train", split)]
    return ds


def _make_fnn_mock() -> MagicMock:
    row = {"id": "1", "title": "Test article", "label": "fake"}
    split = MagicMock()
    split.__iter__ = MagicMock(return_value=iter([row]))
    ds = MagicMock()
    ds.items.return_value = [("train", split)]
    return ds


class _ColumnMock(MagicMock):
    """MagicMock whose comparison operators return MagicMock instead of NotImplemented."""

    def __gt__(self, other):
        return MagicMock()

    def __lt__(self, other):
        return MagicMock()

    def __ge__(self, other):
        return MagicMock()

    def __le__(self, other):
        return MagicMock()


def _make_pyspark_modules() -> dict:
    """Inject fake sys.modules for PySpark/Delta so lazy imports inside functions resolve."""
    mock_dt = MagicMock()
    mock_dt.isDeltaTable.return_value = False

    col_mock = _ColumnMock()
    mock_f = MagicMock()
    for attr in ("col", "length", "trim", "regexp_replace", "hash", "pmod"):
        getattr(mock_f, attr).return_value = col_mock

    modules = {
        "pyspark": MagicMock(),
        "pyspark.sql": MagicMock(),
        "pyspark.sql.functions": mock_f,
        "pyspark.sql.types": MagicMock(),
        "delta": MagicMock(),
        "delta.tables": MagicMock(),
    }
    modules["delta.tables"].DeltaTable = mock_dt
    return modules


@pytest.mark.unit
class TestRunLoadsDatasets:
    def test_run_calls_load_dataset_for_both_sources(self):
        from src.ingestion.ingest_static import run

        with (
            patch("src.ingestion.ingest_static.load_dataset") as mock_load,
            patch("src.ingestion.ingest_static._merge_to_bronze"),
        ):
            mock_load.side_effect = [_make_liar_mock(), _make_fnn_mock()]
            run()

        assert mock_load.call_count == 2
        mock_load.assert_any_call("liar", trust_remote_code=True)
        mock_load.assert_any_call("rickstello/FakeNewsNet", trust_remote_code=True)

    def test_run_calls_merge_with_combined_records(self):
        from src.ingestion.ingest_static import run

        with (
            patch("src.ingestion.ingest_static.load_dataset") as mock_load,
            patch("src.ingestion.ingest_static._merge_to_bronze") as mock_merge,
        ):
            mock_load.side_effect = [_make_liar_mock(), _make_fnn_mock()]
            run()

        mock_merge.assert_called_once()
        all_records = mock_merge.call_args[0][0]
        assert len(all_records) == 2  # 1 liar + 1 fnn row from mocks
        sources = {r["source"] for r in all_records}
        assert sources == {"liar", "fakenewsnet"}

    def test_run_completes_without_error(self):
        from src.ingestion.ingest_static import run

        with (
            patch("src.ingestion.ingest_static.load_dataset") as mock_load,
            patch("src.ingestion.ingest_static._merge_to_bronze"),
        ):
            mock_load.side_effect = [_make_liar_mock(), _make_fnn_mock()]
            run()  # must not raise


@pytest.mark.unit
class TestMergeToBronze:
    def test_calls_get_spark_once(self):
        from src.ingestion.ingest_static import _merge_to_bronze

        mock_spark = MagicMock()
        with (
            patch.dict(sys.modules, _make_pyspark_modules()),
            patch("src.spark_session.get_spark", return_value=mock_spark) as mock_get_spark,
        ):
            _merge_to_bronze([], "/tmp/bronze")

        mock_get_spark.assert_called_once()

    def test_calls_create_dataframe_with_bronze_columns(self):
        from src.ingestion.ingest_static import _merge_to_bronze

        mock_spark = MagicMock()
        records = [
            {"id": "1", "text": "claim", "label": "true", "source": "liar", "raw_meta": "{}"}
        ]

        with (
            patch.dict(sys.modules, _make_pyspark_modules()),
            patch("src.spark_session.get_spark", return_value=mock_spark),
        ):
            _merge_to_bronze(records, "/tmp/bronze")

        mock_spark.createDataFrame.assert_called_once()
        passed_records = mock_spark.createDataFrame.call_args[0][0]
        assert passed_records == records
        assert set(passed_records[0].keys()) == {"id", "text", "label", "source", "raw_meta"}

    def test_adds_ingested_at_column(self):
        from src.ingestion.ingest_static import _merge_to_bronze

        mock_spark = MagicMock()
        with (
            patch.dict(sys.modules, _make_pyspark_modules()),
            patch("src.spark_session.get_spark", return_value=mock_spark),
        ):
            _merge_to_bronze([], "/tmp/bronze")

        mock_spark.createDataFrame.return_value.withColumn.assert_called_once()
        col_name = mock_spark.createDataFrame.return_value.withColumn.call_args[0][0]
        assert col_name == "ingested_at"

    def test_writes_new_table_when_delta_table_does_not_exist(self):
        from src.ingestion.ingest_static import _merge_to_bronze

        mock_spark = MagicMock()
        mods = _make_pyspark_modules()
        mods["delta.tables"].DeltaTable.isDeltaTable.return_value = False

        with (
            patch.dict(sys.modules, mods),
            patch("src.spark_session.get_spark", return_value=mock_spark),
        ):
            _merge_to_bronze([], "/tmp/bronze")

        mock_spark.createDataFrame.return_value.withColumn.return_value.write.format.assert_called_once_with(
            "delta"
        )

    def test_uses_delta_merge_when_table_exists(self):
        from src.ingestion.ingest_static import _merge_to_bronze

        mock_spark = MagicMock()
        mods = _make_pyspark_modules()
        mods["delta.tables"].DeltaTable.isDeltaTable.return_value = True

        with (
            patch.dict(sys.modules, mods),
            patch("src.spark_session.get_spark", return_value=mock_spark),
        ):
            _merge_to_bronze([], "/tmp/bronze")

        mods["delta.tables"].DeltaTable.forPath.assert_called_once_with(mock_spark, "/tmp/bronze")
