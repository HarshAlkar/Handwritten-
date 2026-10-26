"""Tests for model loading and prediction shape/probability constraints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ML.model.architecture import NUM_CLASSES, build_digit_cnn
from ML.training.config import TrainConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "ML" / "artifacts" / "digit_cnn.keras"


def test_build_model_output_shape():
    model = build_digit_cnn()
    x = np.random.rand(2, 28, 28, 1).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (2, NUM_CLASSES)
    assert np.all(y >= 0.0) and np.all(y <= 1.0)
    np.testing.assert_allclose(y.sum(axis=1), np.ones(2), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Trained model artifact not found")
def test_trained_model_probabilities():
    from tensorflow import keras

    model = keras.models.load_model(MODEL_PATH)
    x = np.zeros((1, 28, 28, 1), dtype=np.float32)
    # synthetic white stroke
    x[0, 4:24, 12:16, 0] = 1.0
    probs = model.predict(x, verbose=0)[0]
    assert probs.shape == (NUM_CLASSES,)
    assert float(probs.min()) >= 0.0
    assert float(probs.max()) <= 1.0
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert 0 <= int(np.argmax(probs)) <= 9


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.dataset_name == "ylecun/mnist"
    assert cfg.random_seed == 42
