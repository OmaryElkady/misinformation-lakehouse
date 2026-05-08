"""
tests/unit/test_ingestion.py

Unit tests for the ingestion layer.
No Spark, no network — all external calls are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import patch

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
        mock_load.assert_called_once_with("liar")

    def test_run_raises_when_fakenewsnet_dataset_fails_to_load(self):
        import unittest.mock as mock

        from src.ingestion.ingest_static import run

        # Minimal mock of the LIAR dataset structure

        liar_label_feature = mock.MagicMock()
        liar_label_feature.names = ["false", "half-true", "mostly-true", "true", "barely-true", "pants-fire"]

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
