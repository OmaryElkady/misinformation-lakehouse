"""
tests/unit/test_api_health.py

Liveness probe tests for the FastAPI serving layer.
The /health endpoint must always return 200 — this is what
Kubernetes / Docker healthchecks rely on.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.serving.app import app


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_returns_200():
    with patch("src.serving.app.model_loader") as mock_loader:
        mock_loader.loaded = False
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_returns_ok_status():
    with patch("src.serving.app.model_loader") as mock_loader:
        mock_loader.loaded = False
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
