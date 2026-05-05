"""
train.py — Fine-tune RoBERTa on Gold table features, track with MLflow.

Model: roberta-base fine-tuned for sequence classification (2 classes)
Training data: Gold Delta table (train split)
MLflow artifacts: model weights, tokenizer, confusion matrix, ROC curve

TODO: Implement in Step 4 (Training Layer)
"""

from __future__ import annotations


def run(experiment_name: str | None = None) -> str:
    """Returns the MLflow run_id of the completed training run."""
    raise NotImplementedError("Implement in Step 4")
