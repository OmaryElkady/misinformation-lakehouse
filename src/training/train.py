"""
train.py — Fine-tune RoBERTa on Gold table features, track with MLflow.

Workflow (two-phase, because training runs on Google Colab):
  Phase 1 (local):  export_gold_to_parquet() → writes train/val parquets
  Phase 2 (Colab):  train() reads those parquets, fine-tunes, logs to MLflow
                    via ngrok tunnel pointing at the local tracking server

Model: roberta-base, sequence classification (2 classes: 0=credible, 1=misinfo)
Registry name: settings.model_name  ("misinformation-roberta-v1")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

# ── Training hyper-parameters ──────────────────────────────────────────────────
MODEL_CHECKPOINT = "roberta-base"
MAX_LENGTH = 128
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


# ── Pure helpers ───────────────────────────────────────────────────────────────


def compute_metrics(eval_pred) -> dict:
    """Pure function: EvalPrediction → classification metric dict.

    Accepts any iterable that unpacks to (logits, labels) so unit tests
    can pass a plain (np.ndarray, np.ndarray) tuple without importing
    transformers.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
    }


# ── Phase 1: export Gold → parquet ────────────────────────────────────────────


def export_gold_to_parquet() -> tuple[str, str]:
    """Read Gold Delta table and export train/val splits to parquet files.

    Returns (train_path, val_path) so the caller knows where to find them.
    The files are written to settings.gold_export_path.
    """
    from src.config import settings
    from src.spark_session import get_spark

    spark = get_spark("ExportGold")
    gold_path = settings.delta_path("gold")
    export_dir = settings.gold_export_path

    Path(export_dir).mkdir(parents=True, exist_ok=True)

    gold_df = spark.read.format("delta").load(gold_path)

    train_path = f"{export_dir}/train.parquet"
    val_path = f"{export_dir}/val.parquet"

    cols = ["text_clean", "label_binary"]

    train_pd = gold_df.filter(gold_df["split"] == "train").select(*cols).toPandas()
    train_pd.to_parquet(train_path, index=False)
    logger.info(f"Exported {len(train_pd):,} training rows → {train_path}")

    val_pd = gold_df.filter(gold_df["split"] == "val").select(*cols).toPandas()
    val_pd.to_parquet(val_path, index=False)
    logger.info(f"Exported {len(val_pd):,} val rows → {val_path}")

    return train_path, val_path


# ── Phase 2: fine-tune + MLflow ───────────────────────────────────────────────


def train(experiment_name: str | None = None) -> str:
    """Fine-tune roberta-base on exported parquets and log everything to MLflow.

    Reads from settings.gold_export_path (train.parquet / val.parquet).
    Connects to MLflow at settings.mlflow_tracking_uri — point that at your
    ngrok URL when running from Colab.

    Returns the MLflow run_id of the completed run.
    """
    import mlflow
    import mlflow.transformers
    import pandas as pd
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from src.config import settings

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name or settings.mlflow_experiment_name)

    export_dir = settings.gold_export_path
    train_df = pd.read_parquet(f"{export_dir}/train.parquet")
    val_df = pd.read_parquet(f"{export_dir}/val.parquet")

    logger.info(f"Loaded {len(train_df):,} training rows, {len(val_df):,} val rows")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def _tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text_clean"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    train_ds = (
        Dataset.from_pandas(train_df)
        .rename_column("label_binary", "labels")
        .map(_tokenize, batched=True)
    )
    val_ds = (
        Dataset.from_pandas(val_df)
        .rename_column("label_binary", "labels")
        .map(_tokenize, batched=True)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=2
    )

    output_dir = "./data/models/roberta-v1"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir="./logs",
        report_to="none",  # MLflow is handled manually below
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model_name": MODEL_CHECKPOINT,
                "learning_rate": LEARNING_RATE,
                "num_epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "training_samples": len(train_df),
                "val_samples": len(val_df),
            }
        )

        trainer.train()

        # Log eval metrics once per epoch
        for entry in trainer.state.log_history:
            if "eval_f1" not in entry:
                continue
            epoch = int(entry.get("epoch", 0))
            mlflow.log_metrics(
                {
                    "eval_loss": entry["eval_loss"],
                    "eval_f1": entry["eval_f1"],
                    "eval_accuracy": entry["eval_accuracy"],
                    "eval_precision": entry["eval_precision"],
                    "eval_recall": entry["eval_recall"],
                },
                step=epoch,
            )

        # Log model + tokenizer as a single MLflow artifact
        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            artifact_path="model",
            task="text-classification",
        )

        _log_confusion_matrix(trainer, val_ds)

        run_id = run.info.run_id

    logger.info(f"Training complete — MLflow run_id: {run_id}")
    return run_id


def _log_confusion_matrix(trainer, val_ds) -> None:
    """Predict on val set, save confusion matrix PNG, log as MLflow artifact."""
    import matplotlib.pyplot as plt
    import mlflow
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    preds_output = trainer.predict(val_ds)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_true = preds_output.label_ids

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["credible", "misinfo"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Validation Confusion Matrix")

    cm_path = "./data/exports/confusion_matrix.png"
    Path(cm_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(cm_path)
    logger.info(f"Confusion matrix saved → {cm_path}")


# ── Model registry ─────────────────────────────────────────────────────────────


def register_model(run_id: str) -> None:
    """Register the model from run_id in MLflow Model Registry.

    Promotes to 'Production' if eval_f1 >= 0.80, otherwise 'Staging'.
    """
    import mlflow
    from mlflow import MlflowClient

    from src.config import settings

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, settings.model_name)
    version = result.version

    run = client.get_run(run_id)
    eval_f1 = run.data.metrics.get("eval_f1", 0.0)

    if eval_f1 >= 0.80:
        client.transition_model_version_stage(
            name=settings.model_name,
            version=version,
            stage="Production",
        )
        logger.info(
            f"Model '{settings.model_name}' v{version} → Production (eval_f1={eval_f1:.4f})"
        )
    else:
        client.transition_model_version_stage(
            name=settings.model_name,
            version=version,
            stage="Staging",
        )
        logger.info(
            f"Model '{settings.model_name}' v{version} → Staging "
            f"(eval_f1={eval_f1:.4f} < 0.80 threshold)"
        )
