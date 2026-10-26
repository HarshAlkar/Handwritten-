"""Train the digit CNN on Hugging Face MNIST (ylecun/mnist)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow import keras

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ML.model.architecture import build_digit_cnn  # noqa: E402
from ML.preprocessing.pipeline import normalize_array  # noqa: E402
from ML.training.config import ARTIFACTS_DIR, TrainConfig  # noqa: E402


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def hf_split_to_arrays(split) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Hugging Face MNIST split to NumPy arrays."""
    images = []
    labels = []
    for row in split:
        img = np.asarray(row["image"], dtype=np.float32)
        if img.ndim == 3:
            img = img[:, :, 0]
        images.append(img)
        labels.append(int(row["label"]))
    x = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=np.int32)
    return x, y


def prepare_images(x: np.ndarray) -> np.ndarray:
    """Normalize and reshape to (N, 28, 28, 1). HF MNIST is already white-on-black."""
    x = normalize_array(x)
    if x.ndim == 3:
        x = x[..., np.newaxis]
    return x


def load_mnist_arrays(dataset_name: str, seed: int, val_split: float):
    print(f"Loading dataset '{dataset_name}' from Hugging Face...")
    ds = load_dataset(dataset_name)
    x_train_full, y_train_full = hf_split_to_arrays(ds["train"])
    x_test, y_test = hf_split_to_arrays(ds["test"])

    x_train_full = prepare_images(x_train_full)
    x_test = prepare_images(x_test)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(x_train_full))
    rng.shuffle(indices)
    x_train_full = x_train_full[indices]
    y_train_full = y_train_full[indices]

    n_val = int(len(x_train_full) * val_split)
    x_val = x_train_full[:n_val]
    y_val = y_train_full[:n_val]
    x_train = x_train_full[n_val:]
    y_train = y_train_full[n_val:]

    print(
        f"Splits — train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)}"
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def train(cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TrainConfig.from_env()
    set_seeds(cfg.random_seed)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_arrays(
        cfg.dataset_name, cfg.random_seed, cfg.validation_split
    )

    model = build_digit_cnn(learning_rate=cfg.learning_rate)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            mode="max",
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(cfg.checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(cfg.model_path)
    print(f"Saved model to {cfg.model_path}")

    history_payload = {
        "config": cfg.to_dict(),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "epochs_trained": len(history.history.get("loss", [])),
        "dataset": cfg.dataset_name,
        "split_sizes": {
            "train": int(len(x_train)),
            "validation": int(len(x_val)),
            "test": int(len(x_test)),
        },
    }
    with open(cfg.history_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)
    print(f"Saved training history to {cfg.history_path}")

    # Quick smoke eval on test set (full metrics via evaluate.py)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy (quick): {test_acc:.4f}, loss: {test_loss:.4f}")
    history_payload["quick_test"] = {
        "accuracy": float(test_acc),
        "loss": float(test_loss),
    }
    with open(cfg.history_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    return history_payload


def main() -> None:
    train()


if __name__ == "__main__":
    main()
