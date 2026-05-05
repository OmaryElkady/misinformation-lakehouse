# Real-Time Misinformation Detection Lakehouse Platform

> **Status:** 🚧 In Development  
> **Author:** [Your Name] · Georgia State University, Honors College  
> **Stack:** PySpark · Delta Lake · MLflow · RoBERTa · Llama 3 · FastAPI · Prefect · Docker

---

## Abstract

*(To be written at project completion — Step 8)*

---

## Architecture

*(Architecture diagram to be added — see `docs/architecture.png`)*

### Data Flow

```
Reddit API ──┐
             ├──► Bronze (raw) ──► Silver (clean) ──► Gold (features)
LIAR Dataset─┘         Delta Lake (local / S3)              │
                                                             ▼
                                                    RoBERTa Fine-tune
                                                    + MLflow Registry
                                                             │
                                                             ▼
                                                    FastAPI /predict
                                                    (+ Groq LLM explain)
```

---

## Quickstart

### Prerequisites
- WSL2 (Ubuntu 22.04) on Windows, or macOS/Linux
- Docker Desktop
- Python 3.11+
- (Optional) AWS account for S3 storage

### 1. Clone & configure
```bash
git clone https://github.com/YOUR_USERNAME/misinformation-lakehouse
cd misinformation-lakehouse
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start services
```bash
docker compose up -d
# MLflow UI → http://localhost:5000
# Prefect UI → http://localhost:4200
# API docs   → http://localhost:8000/docs
```

### 3. Run the pipeline
```bash
# Ingest datasets
python -m src.ingestion.ingest_static

# Process Bronze → Silver → Gold
python -m src.processing.bronze_to_silver
python -m src.processing.silver_to_gold

# Train model
python -m src.training.train

# The API auto-loads the Production model from MLflow
```

---

## Project Structure

```
misinformation-lakehouse/
├── src/
│   ├── ingestion/         # LIAR/FakeNewsNet loader + Reddit API → Bronze
│   ├── processing/        # Bronze → Silver → Gold PySpark jobs
│   ├── training/          # RoBERTa fine-tune + MLflow experiment tracking
│   ├── serving/           # FastAPI inference server + Dockerfile
│   └── orchestration/     # Prefect pipeline flows
├── notebooks/             # EDA and experimentation
├── tests/
│   ├── unit/              # Fast tests, no external deps (run in CI)
│   └── integration/       # Spark + network tests (run locally)
├── docs/                  # Architecture diagrams, write-ups
├── scripts/               # validate_model.py, setup scripts
├── .github/workflows/     # CI/CD pipeline (lint → test → docker → model gate)
├── docker-compose.yml     # MLflow + FastAPI + Prefect stack
├── requirements.txt
└── pyproject.toml
```

---

## Results

*(Model performance metrics, confusion matrix, latency benchmarks — to be added)*

---

## Design Decisions

*(To be written — Step 8)*

---

## Future Work

*(To be written — Step 8)*

---

## References

- Wang, W. Y. (2017). "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection. *ACL 2017*.
- Shu, K. et al. (2020). FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media. *Big Data*.
- Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv:1907.11692*.
