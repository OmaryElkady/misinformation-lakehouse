FROM python:3.12-slim

# ─── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── App setup ────────────────────────────────────────────────────────────────
WORKDIR /app

# Install only serving-layer deps (keeps image lean)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    pydantic==2.7.1 \
    mlflow==2.13.0 \
    transformers==4.41.2 \
    torch==2.3.0 \
    groq==0.9.0 \
    python-dotenv==1.0.1 \
    loguru==0.7.2

# Copy source
COPY src/ ./src/

# ─── Non-root user for security ───────────────────────────────────────────────
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
