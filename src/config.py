"""
config.py — Central configuration loader.

All environment variables are read here. Import this module everywhere
instead of calling os.getenv() scattered across the codebase.
"""

from __future__ import annotations

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env if present (no-op in Docker / CI where vars are injected)
load_dotenv()


class Settings(BaseSettings):
    # ── Bluesky ───────────────────────────────────────────────────────────────
    bluesky_handle: str = ""
    bluesky_app_password: str = ""

    # ── Reddit ────────────────────────────────────────────────────────────────
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "misinformation-lakehouse/1.0"

    # ── Groq (LLM inference) ─────────────────────────────────────────────────
    groq_api_key: str = ""

    # ── AWS ───────────────────────────────────────────────────────────────────
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "us-east-1"
    s3_bucket_name: str = "misinformation-lakehouse-dev"

    # ── Storage ───────────────────────────────────────────────────────────────
    storage_mode: str = "local"  # "local" | "s3"
    local_delta_path: str = "./data/delta"

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "misinformation-detection"

    # ── FastAPI ───────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_name: str = "misinformation-roberta-v1"

    # ── Prefect ───────────────────────────────────────────────────────────────
    prefect_api_url: str = "http://localhost:4200/api"

    # ── ngrok (expose local MLflow to Google Colab) ───────────────────────────
    ngrok_authtoken: str = ""

    # ── Training exports ──────────────────────────────────────────────────────
    gold_export_path: str = "./data/exports"

    # ── Derived paths (not from env) ──────────────────────────────────────────
    @property
    def bronze_path(self) -> str:
        return f"{self.local_delta_path}/bronze"

    @property
    def silver_path(self) -> str:
        return f"{self.local_delta_path}/silver"

    @property
    def gold_path(self) -> str:
        return f"{self.local_delta_path}/gold"

    @property
    def s3_bronze_path(self) -> str:
        return f"s3a://{self.s3_bucket_name}/delta/bronze"

    @property
    def s3_silver_path(self) -> str:
        return f"s3a://{self.s3_bucket_name}/delta/silver"

    @property
    def s3_gold_path(self) -> str:
        return f"s3a://{self.s3_bucket_name}/delta/gold"

    def delta_path(self, layer: str) -> str:
        """Return the right path for a layer based on STORAGE_MODE."""
        paths = {
            "local": {
                "bronze": self.bronze_path,
                "silver": self.silver_path,
                "gold": self.gold_path,
            },
            "s3": {
                "bronze": self.s3_bronze_path,
                "silver": self.s3_silver_path,
                "gold": self.s3_gold_path,
            },
        }
        return paths[self.storage_mode][layer]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=(),
    )


# ─── Singleton — import this everywhere ───────────────────────────────────────
settings = Settings()
