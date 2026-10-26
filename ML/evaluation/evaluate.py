"""Evaluate a trained digit CNN on the Hugging Face MNIST test split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ML.training.config import ARTIFACTS_DIR, TrainConfig  # noqa: E402
from ML.training.train import hf_split_to_arrays, prepare_images  # noqa: E402


def evaluate(
    model_path: Path | None = None,
    metrics_path: Path | None = None,
    dataset_name: str = "ylecun/mnist",
) -> dict:
    cfg = TrainConfig.from_env()
    model_path = model_path or cfg.model_path
    metrics_path = metrics_path or (ARTIFACTS_DIR / "metrics.json")

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first: python -m ML.training.train"
        )

    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)

    print(f"Loading test split from '{dataset_name}'...")
    ds = load_dataset(dataset_name)
    x_test, y_test = hf_split_to_arrays(ds["test"])
    x_test = prepare_images(x_test)

    loss, keras_acc = model.evaluate(x_test, y_test, verbose=0)
    probs = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    per_class = classification_report(
        y_test, y_pred, digits=4, output_dict=True, zero_division=0
    )

    metrics = {
        "dataset": dataset_name,
        "model_path": str(model_path),
        "test_samples": int(len(y_test)),
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "keras_test_accuracy": float(keras_acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": cm,
        "per_class": {
            str(k): v
            for k, v in per_class.items()
            if k not in {"accuracy", "macro avg", "weighted avg"}
        },
        "macro_avg": per_class.get("macro avg"),
        "weighted_avg": per_class.get("weighted avg"),
    }

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print(f"Saved metrics to {metrics_path}")
    return metrics


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
