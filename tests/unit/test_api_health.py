"""
tests/unit/test_api_health.py

Tests for the FastAPI serving layer.
The /health endpoint must always return 200 — this is what
Kubernetes / Docker healthchecks rely on.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from src.serving.app import app


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_returns_200():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_returns_ok_status():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.json() == {"status": "ok"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_predict_stub_raises_500():
    """Predict is not yet implemented — should return 500, not hang."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/predict", json={"text": "test claim"})
    assert response.status_code == 500
