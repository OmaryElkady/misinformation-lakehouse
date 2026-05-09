from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.serving.app import app
from src.serving.model_loader import ModelLoader

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_unloaded_model():
    with patch("src.serving.app.model_loader") as mock_loader:
        mock_loader.loaded = False
        mock_loader.model_version = "not_loaded"
        mock_loader.run_id = None
        mock_loader.stage = "not_loaded"
        yield mock_loader


@pytest.fixture
def mock_loaded_model():
    with patch("src.serving.app.model_loader") as mock_loader:
        mock_loader.loaded = True
        mock_loader.model_version = "3"
        mock_loader.run_id = "abc123run456"
        mock_loader.stage = "Production"
        mock_loader.predict.return_value = ("misinformation", 0.92)
        yield mock_loader


@pytest.fixture
def mock_groq_client():
    with patch("src.serving.app.groq_client") as mock_client:
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = (
            "This claim appears to be misinformation based on available evidence."
        )
        mock_client.chat.completions.create.return_value = mock_completion
        yield mock_client


# ── GET /health ───────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_always_200(mock_unloaded_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_model_not_loaded(mock_unloaded_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.json() == {"status": "ok", "model_loaded": False}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_model_loaded(mock_loaded_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.json() == {"status": "ok", "model_loaded": True}


# ── GET /model/info ───────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_model_info_returns_200(mock_loaded_model):
    with patch("src.serving.app.MlflowClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.data.metrics = {"eval_f1": 0.85}
        mock_client.get_run.return_value = mock_run
        mock_client_cls.return_value = mock_client

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/model/info")

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "misinformation-roberta-v1"
    assert data["model_version"] == "3"
    assert data["stage"] == "Production"
    assert data["mlflow_run_id"] == "abc123run456"
    assert data["eval_f1"] == pytest.approx(0.85)
    assert data["loaded"] is True


# ── POST /predict ─────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_predict_503_when_not_loaded(mock_unloaded_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json={"text": "A valid test claim here"})
    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"].lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_predict_200_when_loaded(mock_loaded_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json={"text": "A valid test claim here"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "misinformation"
    assert data["explanation"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_predict_confidence_in_range(mock_loaded_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json={"text": "Some verifiable claim text"})
    assert 0.0 <= response.json()["confidence"] <= 1.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_predict_latency_positive(mock_loaded_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json={"text": "Some verifiable claim text"})
    assert response.json()["latency_ms"] > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_predict_no_explain_does_not_call_groq(mock_loaded_model, mock_groq_client):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/predict", json={"text": "Some verifiable claim text", "explain": False}
        )
    assert response.status_code == 200
    mock_groq_client.chat.completions.create.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_predict_explain_returns_non_null_explanation(mock_loaded_model, mock_groq_client):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/predict", json={"text": "Some verifiable claim text", "explain": True}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["explanation"] is not None
    assert isinstance(data["explanation"], str)
    mock_groq_client.chat.completions.create.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_predict_short_text_422(mock_unloaded_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json={"text": "short"})
    assert response.status_code == 422


@pytest.mark.unit
@pytest.mark.asyncio
async def test_predict_long_text_422(mock_unloaded_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json={"text": "x" * 2001})
    assert response.status_code == 422


# ── POST /predict/batch ───────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_returns_200_with_results(mock_loaded_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/predict/batch", json={"texts": ["This is a test claim for batch prediction"]}
        )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["total_latency_ms"] > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_empty_list_422(mock_unloaded_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_too_many_items_422(mock_unloaded_model):
    texts = ["A valid claim here that is long enough" for _ in range(33)]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 422


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_total_latency_positive(mock_loaded_model):
    texts = ["First claim for batch testing", "Second claim for batch testing"]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict/batch", json={"texts": texts})
    assert response.json()["total_latency_ms"] > 0


# ── ModelLoader unit tests (no network) ──────────────────────────────────────


@pytest.mark.unit
def test_model_loader_predict_raises_when_not_loaded():
    loader = ModelLoader()
    with pytest.raises(RuntimeError, match="not loaded"):
        loader.predict("some text to classify here")


@pytest.mark.unit
def test_model_loader_predict_returns_tuple_when_loaded():
    loader = ModelLoader()
    loader.loaded = True
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"label": "LABEL_0", "score": 0.92}]
    loader.model = mock_pipeline

    label, confidence = loader.predict("some text to classify here")
    assert isinstance(label, str)
    assert label in ("misinformation", "credible", "unknown")
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


@pytest.mark.unit
def test_model_loader_load_sets_loaded_true():
    loader = ModelLoader()

    mock_client = MagicMock()
    mock_version = MagicMock()
    mock_version.version = "3"
    mock_version.run_id = "abc123"
    mock_client.get_latest_versions.return_value = [mock_version]

    mock_mlflow = MagicMock()
    mock_mlflow.transformers.load_model.return_value = MagicMock()

    mock_tracking = MagicMock()
    mock_tracking.MlflowClient.return_value = mock_client

    # Inject mocks via sys.modules so the lazy `import mlflow` inside load() gets them.
    with patch.dict(sys.modules, {"mlflow": mock_mlflow, "mlflow.tracking": mock_tracking}):
        loader.load()

    assert loader.loaded is True


@pytest.mark.unit
def test_model_loader_load_does_not_raise_on_mlflow_error():
    loader = ModelLoader()

    mock_tracking = MagicMock()
    mock_tracking.MlflowClient.side_effect = Exception("Connection refused to MLflow")

    with patch.dict(sys.modules, {"mlflow": MagicMock(), "mlflow.tracking": mock_tracking}):
        loader.load()  # must not raise

    assert loader.loaded is False
