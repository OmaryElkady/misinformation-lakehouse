"""
tests/unit/test_training.py

Unit tests for the training layer.
No Spark, no network, no real model — all external calls are mocked.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── compute_metrics ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeMetrics:
    def test_perfect_predictions_give_f1_one(self):
        from src.training.train import compute_metrics

        logits = np.array([[10.0, -10.0], [10.0, -10.0], [-10.0, 10.0]])
        labels = np.array([0, 0, 1])
        result = compute_metrics((logits, labels))
        assert result["f1"] == pytest.approx(1.0)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_all_wrong_predictions_give_low_f1(self):
        from src.training.train import compute_metrics

        # model predicts class 1 for everything, truth is class 0 for everything
        logits = np.array([[-10.0, 10.0]] * 4)
        labels = np.array([0, 0, 0, 0])
        result = compute_metrics((logits, labels))
        # All predictions wrong → f1 = 0 (zero_division=0 guard)
        assert result["f1"] == pytest.approx(0.0)
        assert result["accuracy"] == pytest.approx(0.0)

    def test_mixed_predictions_give_intermediate_f1(self):
        from src.training.train import compute_metrics

        # 3 correct out of 6 → f1 between 0 and 1
        logits = np.array(
            [
                [10.0, -10.0],  # pred 0, true 0 ✓
                [10.0, -10.0],  # pred 0, true 0 ✓
                [10.0, -10.0],  # pred 0, true 1 ✗
                [-10.0, 10.0],  # pred 1, true 1 ✓
                [-10.0, 10.0],  # pred 1, true 0 ✗
                [-10.0, 10.0],  # pred 1, true 0 ✗
            ]
        )
        labels = np.array([0, 0, 1, 1, 0, 0])
        result = compute_metrics((logits, labels))
        assert 0.0 < result["f1"] < 1.0
        assert 0.0 < result["accuracy"] < 1.0

    def test_returns_all_required_keys(self):
        from src.training.train import compute_metrics

        logits = np.array([[10.0, -10.0], [-10.0, 10.0]])
        labels = np.array([0, 1])
        result = compute_metrics((logits, labels))
        assert set(result.keys()) == {"f1", "accuracy", "precision", "recall"}

    def test_is_pure_with_no_side_effects(self):
        from src.training.train import compute_metrics

        logits = np.array([[10.0, -10.0], [-10.0, 10.0]])
        labels = np.array([0, 1])
        result1 = compute_metrics((logits, labels))
        result2 = compute_metrics((logits, labels))
        assert result1 == result2


# ── register_model ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRegisterModel:
    def _make_mock_client(self, eval_f1: float) -> MagicMock:
        mock_run = MagicMock()
        mock_run.data.metrics = {"eval_f1": eval_f1}
        client = MagicMock()
        client.get_run.return_value = mock_run
        return client

    def _run(self, mock_client: MagicMock, version: str = "3") -> None:
        from src.training.train import register_model

        mock_result = MagicMock()
        mock_result.version = version

        mock_mlflow = MagicMock()
        mock_mlflow.register_model.return_value = mock_result
        mock_mlflow.MlflowClient.return_value = mock_client

        with patch.dict(sys.modules, {"mlflow": mock_mlflow, "mlflow.tracking": MagicMock()}):
            register_model("fake-run-id")

    def test_promotes_to_production_when_f1_above_threshold(self):
        client = self._make_mock_client(eval_f1=0.85)
        self._run(client, version="2")
        client.set_registered_model_alias.assert_called_once_with(
            "misinformation-roberta-v1", "production", "2"
        )

    def test_does_not_promote_to_production_when_f1_below_threshold(self):
        client = self._make_mock_client(eval_f1=0.75)
        self._run(client)
        calls = [str(c) for c in client.set_registered_model_alias.call_args_list]
        assert not any("production" in c for c in calls)

    def test_sends_to_staging_when_f1_below_threshold(self):
        client = self._make_mock_client(eval_f1=0.75)
        self._run(client, version="1")
        client.set_registered_model_alias.assert_called_once_with(
            "misinformation-roberta-v1", "staging", "1"
        )

    def test_promotes_at_exact_threshold(self):
        # eval_f1 == 0.80 should still promote to Production
        client = self._make_mock_client(eval_f1=0.80)
        self._run(client, version="4")
        client.set_registered_model_alias.assert_called_once_with(
            "misinformation-roberta-v1", "production", "4"
        )

    def test_registers_model_with_correct_uri(self):
        client = self._make_mock_client(eval_f1=0.85)
        mock_result = MagicMock()
        mock_result.version = "1"

        mock_mlflow = MagicMock()
        mock_mlflow.register_model.return_value = mock_result
        mock_mlflow.MlflowClient.return_value = client

        with patch.dict(sys.modules, {"mlflow": mock_mlflow, "mlflow.tracking": MagicMock()}):
            from src.training.train import register_model

            register_model("my-run-id")

        mock_mlflow.register_model.assert_called_once_with(
            "runs:/my-run-id/model", "misinformation-roberta-v1"
        )


# ── export_gold_to_parquet ─────────────────────────────────────────────────────


def _make_pyspark_modules() -> dict:
    modules = {
        "pyspark": MagicMock(),
        "pyspark.sql": MagicMock(),
        "pyspark.sql.functions": MagicMock(),
        "pyspark.sql.types": MagicMock(),
        "delta": MagicMock(),
        "delta.tables": MagicMock(),
    }
    return modules


@pytest.mark.unit
class TestExportGoldToParquet:
    def _make_spark_mock(self, train_rows: int = 100, val_rows: int = 20):
        import pandas as pd

        train_pd = pd.DataFrame(
            {"text_clean": ["claim"] * train_rows, "label_binary": [0] * train_rows}
        )
        val_pd = pd.DataFrame({"text_clean": ["claim"] * val_rows, "label_binary": [1] * val_rows})

        mock_df = MagicMock()
        # filter returns mock_df, select returns it, toPandas returns real DataFrames in order
        mock_df.filter.return_value = mock_df
        mock_df.select.return_value = mock_df
        mock_df.toPandas.side_effect = [train_pd, val_pd]

        mock_spark = MagicMock()
        mock_spark.read.format.return_value.load.return_value = mock_df

        return mock_spark

    # All tests mock pandas.DataFrame.to_parquet to avoid a pyarrow version
    # conflict in the CI environment.  We verify the call arguments instead
    # of checking files on disk.

    def _ctx(self, tmp_path):
        """Return the common set of context-manager patches as a tuple."""
        mock_settings = MagicMock()
        mock_settings.delta_path.return_value = "/fake/gold"
        mock_settings.gold_export_path = str(tmp_path)
        mock_spark = self._make_spark_mock()
        return mock_settings, mock_spark

    def test_returns_tuple_of_two_strings(self, tmp_path):
        from src.training.train import export_gold_to_parquet

        mock_settings, mock_spark = self._ctx(tmp_path)

        with (
            patch.dict(sys.modules, _make_pyspark_modules()),
            patch("src.spark_session.get_spark", return_value=mock_spark),
            patch("src.config.settings", mock_settings),
            patch("pandas.DataFrame.to_parquet"),
        ):
            result = export_gold_to_parquet()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_writes_parquet_to_correct_paths(self, tmp_path):
        from src.training.train import export_gold_to_parquet

        mock_settings, mock_spark = self._ctx(tmp_path)

        with (
            patch.dict(sys.modules, _make_pyspark_modules()),
            patch("src.spark_session.get_spark", return_value=mock_spark),
            patch("src.config.settings", mock_settings),
            patch("pandas.DataFrame.to_parquet") as mock_to_parquet,
        ):
            train_path, val_path = export_gold_to_parquet()

        assert train_path.endswith("train.parquet")
        assert val_path.endswith("val.parquet")
        # Verify to_parquet was called twice (once for train, once for val)
        assert mock_to_parquet.call_count == 2

    def test_to_parquet_called_with_train_then_val_path(self, tmp_path):
        from src.training.train import export_gold_to_parquet

        mock_settings, mock_spark = self._ctx(tmp_path)

        with (
            patch.dict(sys.modules, _make_pyspark_modules()),
            patch("src.spark_session.get_spark", return_value=mock_spark),
            patch("src.config.settings", mock_settings),
            patch("pandas.DataFrame.to_parquet") as mock_to_parquet,
        ):
            train_path, val_path = export_gold_to_parquet()

        called_paths = [call.args[0] for call in mock_to_parquet.call_args_list]
        assert called_paths[0] == train_path
        assert called_paths[1] == val_path

    def test_reads_gold_delta_table(self, tmp_path):
        from src.training.train import export_gold_to_parquet

        mock_settings, mock_spark = self._ctx(tmp_path)

        with (
            patch.dict(sys.modules, _make_pyspark_modules()),
            patch("src.spark_session.get_spark", return_value=mock_spark),
            patch("src.config.settings", mock_settings),
            patch("pandas.DataFrame.to_parquet"),
        ):
            export_gold_to_parquet()

        mock_spark.read.format.assert_called_with("delta")
        mock_spark.read.format.return_value.load.assert_called_once_with("/fake/gold")

    def test_calls_get_spark_once(self, tmp_path):
        from src.training.train import export_gold_to_parquet

        mock_settings, mock_spark = self._ctx(tmp_path)

        with (
            patch.dict(sys.modules, _make_pyspark_modules()),
            patch("src.spark_session.get_spark", return_value=mock_spark) as mock_get_spark,
            patch("src.config.settings", mock_settings),
            patch("pandas.DataFrame.to_parquet"),
        ):
            export_gold_to_parquet()

        mock_get_spark.assert_called_once()
