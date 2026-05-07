"""
tests/unit/test_config.py

Unit tests for the Settings config module.
These tests run with zero external dependencies (no Spark, no network).
"""

import os
from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestSettings:
    def test_default_storage_mode_is_local(self):
        from src.config import Settings
        s = Settings()
        assert s.storage_mode == "local"

    def test_bronze_path_uses_local_delta_path(self):
        from src.config import Settings
        s = Settings(local_delta_path="/tmp/delta")
        assert s.bronze_path == "/tmp/delta/bronze"
        assert s.silver_path == "/tmp/delta/silver"
        assert s.gold_path == "/tmp/delta/gold"

    def test_delta_path_local_returns_local_paths(self):
        from src.config import Settings
        s = Settings(storage_mode="local", local_delta_path="/tmp/delta")
        assert s.delta_path("bronze") == "/tmp/delta/bronze"
        assert s.delta_path("silver") == "/tmp/delta/silver"
        assert s.delta_path("gold") == "/tmp/delta/gold"

    def test_delta_path_s3_returns_s3_paths(self):
        from src.config import Settings
        s = Settings(storage_mode="s3", s3_bucket_name="my-bucket")
        assert s.delta_path("bronze") == "s3a://my-bucket/delta/bronze"

    def test_s3_paths_use_bucket_name(self):
        from src.config import Settings
        s = Settings(s3_bucket_name="test-bucket")
        assert "test-bucket" in s.s3_bronze_path
        assert "test-bucket" in s.s3_silver_path
        assert "test-bucket" in s.s3_gold_path

    def test_env_var_override(self):
        from src.config import Settings
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://custom:9999"}):
            s = Settings()
            assert s.mlflow_tracking_uri == "http://custom:9999"
