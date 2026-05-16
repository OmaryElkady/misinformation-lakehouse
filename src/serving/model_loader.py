from __future__ import annotations

from loguru import logger

from src.config import settings


class ModelLoader:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_version: str = "not_loaded"
        self.loaded: bool = False
        self.run_id: str | None = None
        self.stage: str = "not_loaded"

    def load(self) -> None:
        # Lazy import: mlflow is heavy and may be unavailable in some envs.
        # Tests inject mocks via patch.dict(sys.modules, ...) before this runs.
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            client = MlflowClient()

            for alias in ("production", "staging"):
                try:
                    version = client.get_model_version_by_alias(settings.model_name, alias)
                except Exception:
                    continue

                model_uri = f"models:/{settings.model_name}@{alias}"
                pipeline = mlflow.transformers.load_model(model_uri)
                self.model = pipeline
                self.model_version = str(version.version)
                self.run_id = version.run_id
                self.stage = alias
                self.loaded = True
                logger.info(
                    f"Loaded {settings.model_name} v{version.version} ({alias}) from MLflow"
                )
                return

            logger.warning(
                f"No 'production' or 'staging' alias found for '{settings.model_name}' "
                "— API starting without a model. /predict will return 503."
            )
        except Exception as exc:
            logger.error(f"Model load failed: {exc}")
            self.loaded = False

    def predict(self, text: str) -> tuple[str, float]:
        if not self.loaded:
            raise RuntimeError("Model is not loaded")

        outputs = self.model(text, truncation=True, max_length=512)
        raw_label: str = outputs[0]["label"].upper()
        score: float = float(outputs[0]["score"])

        if "0" in raw_label:
            label = "misinformation"
        elif "1" in raw_label:
            label = "credible"
        else:
            label = "unknown"

        return label, score

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        return [self.predict(t) for t in texts]
