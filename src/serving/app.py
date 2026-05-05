"""
app.py — FastAPI inference server.

Endpoints:
  GET  /health              liveness probe
  GET  /model/info          current loaded model version + metrics
  POST /predict             classify a single text claim
  POST /predict/batch       classify up to 32 claims

TODO: Implement in Step 5 (Serving Layer)
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="Misinformation Detection API",
    description="Real-time misinformation classification powered by fine-tuned RoBERTa + Llama 3.",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(payload: dict) -> dict:
    raise NotImplementedError("Implement in Step 5")
