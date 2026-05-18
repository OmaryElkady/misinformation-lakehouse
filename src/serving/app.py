from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.config import settings
from src.serving.model_loader import ModelLoader

# groq may not be installed in all environments (e.g. CI minimal install).
try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore[assignment, misc]

# mlflow may not be importable in all environments (e.g. pkg_resources missing).
# Import gracefully so the app still starts; tests patch this name directly.
try:
    from mlflow.tracking import MlflowClient
except Exception:
    MlflowClient = None  # type: ignore[assignment, misc]

# ── Module-level singletons ───────────────────────────────────────────────────

model_loader = ModelLoader()
groq_client = (
    Groq(api_key=settings.groq_api_key) if (settings.groq_api_key and Groq is not None) else None
)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    loop = asyncio.get_running_loop()
    try:
        # run_in_executor keeps the event loop unblocked while mlflow makes
        # synchronous HTTP calls; wait_for guarantees we yield within 15 s
        # even if the tracking server accepts the TCP connection but never
        # responds (the failure mode seen on GitHub Actions runners).
        await asyncio.wait_for(
            loop.run_in_executor(None, model_loader.load),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        logger.warning("Model load timed out after 15 s — starting without a model")
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Misinformation Detection API",
    description="Real-time misinformation classification powered by fine-tuned RoBERTa + Llama 3.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic models ───────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    text: Annotated[str, Field(min_length=10, max_length=2000)]
    explain: bool = False


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    text: str
    label: str
    confidence: float
    explanation: str | None
    model_version: str
    latency_ms: float


class BatchPredictRequest(BaseModel):
    texts: Annotated[list[str], Field(min_length=1, max_length=32)]
    explain: bool = False


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total_latency_ms: float


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    stage: str
    mlflow_run_id: str
    eval_f1: float | None
    loaded: bool


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_explanation(label: str, confidence: float, text: str) -> str | None:
    if groq_client is None:
        logger.warning("Groq client not configured — skipping explanation")
        return None

    prompt = (
        f"You are a misinformation analyst. The following claim was "
        f"classified as {label} with {confidence:.0%} confidence. "
        f"Explain in 2-3 sentences why this claim may or may not be "
        f"misinformation, focusing on verifiable facts.\n"
        f"Claim: {text}"
    )
    completion = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": model_loader.loaded}


@app.post("/reload")
async def reload_model() -> dict:
    model_loader.load()
    return {
        "loaded": model_loader.loaded,
        "model_version": model_loader.model_version,
        "stage": model_loader.stage,
    }


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    eval_f1: float | None = None
    if model_loader.run_id and MlflowClient is not None:
        try:
            client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
            run = client.get_run(model_loader.run_id)
            eval_f1 = run.data.metrics.get("eval_f1")
        except Exception as exc:
            logger.warning(f"Could not fetch run metrics from MLflow: {exc}")

    return ModelInfoResponse(
        model_name=settings.model_name,
        model_version=model_loader.model_version,
        stage=model_loader.stage,
        mlflow_run_id=model_loader.run_id or "not_loaded",
        eval_f1=eval_f1,
        loaded=model_loader.loaded,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    if not model_loader.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded — try again later")

    start = time.perf_counter()
    label, confidence = model_loader.predict(request.text)

    explanation: str | None = None
    if request.explain:
        explanation = _get_explanation(label, confidence, request.text)

    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        text=request.text,
        label=label,
        confidence=confidence,
        explanation=explanation,
        model_version=model_loader.model_version,
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    if not model_loader.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded — try again later")

    total_start = time.perf_counter()
    results: list[PredictResponse] = []

    for text in request.texts:
        item_start = time.perf_counter()
        label, confidence = model_loader.predict(text)

        explanation: str | None = None
        if request.explain:
            explanation = _get_explanation(label, confidence, text)

        item_latency_ms = (time.perf_counter() - item_start) * 1000
        results.append(
            PredictResponse(
                text=text,
                label=label,
                confidence=confidence,
                explanation=explanation,
                model_version=model_loader.model_version,
                latency_ms=item_latency_ms,
            )
        )

    total_latency_ms = (time.perf_counter() - total_start) * 1000
    return BatchPredictResponse(results=results, total_latency_ms=total_latency_ms)
