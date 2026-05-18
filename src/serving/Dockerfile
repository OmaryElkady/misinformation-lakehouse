FROM python:3.12-slim

# ─── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── App setup ────────────────────────────────────────────────────────────────
WORKDIR /app

# Install CPU-only PyTorch first (avoids 3 GB CUDA download in a serving container).
# transformers>=5.0.0 requires torch>=2.4.0; the CPU wheel from download.pytorch.org
# satisfies that without pulling a CUDA build from PyPI.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install only serving-layer deps (keeps image lean)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    pydantic==2.7.1 \
    pydantic-settings==2.3.0 \
    'mlflow>=3.0.0,<4.0.0' \
    'transformers>=5.0.0,<6.0.0' \
    'groq>=0.13.0' \
    python-dotenv==1.0.1 \
    loguru==0.7.2

# Copy source
COPY src/ ./src/

# ─── Non-root user for security ───────────────────────────────────────────────
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Prevent MLflow HTTP calls from hanging for the default 120 s when the
# tracking server is unreachable (e.g. CI). 5 s is plenty for a co-located server.
ENV MLFLOW_HTTP_REQUEST_TIMEOUT=5

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
