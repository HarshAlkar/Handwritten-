"""Model loading and inference service."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from tensorflow import keras

from Backend.app.config import Settings, get_settings
from ML.model.architecture import INPUT_SHAPE, NUM_CLASSES, model_summary_dict
from ML.preprocessing.pipeline import PreprocessError, preprocess_image_bytes

logger = logging.getLogger(__name__)


class PredictorService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: keras.Model | None = None
        self._load_error: str | None = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        path = self.settings.resolved_model_path()
        if not path.exists():
            self._load_error = "Model file not found. Train the model first."
            logger.warning("Model not found at configured path")
            return False
        try:
            self._model = keras.models.load_model(path)
            self._load_error = None
            logger.info("Model loaded successfully")
            return True
        except Exception:  # noqa: BLE001
            self._model = None
            self._load_error = "Failed to load model."
            logger.exception("Failed to load model")
            return False

    def ensure_loaded(self) -> None:
        if self._model is None:
            loaded = self.load()
            if not loaded:
                raise RuntimeError(self._load_error or "Model unavailable.")

    def predict_from_bytes(
        self,
        data: bytes,
        content_type: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        self.ensure_loaded()
        assert self._model is not None

        try:
            tensor = preprocess_image_bytes(data, content_type=content_type, filename=filename)
        except PreprocessError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PreprocessError("Failed to preprocess image.") from exc

        probs = self._model.predict(tensor, verbose=0)[0]
        probs = np.asarray(probs, dtype=np.float64)
        # Numerical safety: clip and renormalize
        probs = np.clip(probs, 0.0, 1.0)
        total = probs.sum()
        if total > 0:
            probs = probs / total

        digit = int(np.argmax(probs))
        confidence = float(probs[digit])
        probabilities = {str(i): float(probs[i]) for i in range(NUM_CLASSES)}

        return {
            "predicted_digit": digit,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": "digit_cnn",
            "input_shape": list(INPUT_SHAPE),
            "num_classes": NUM_CLASSES,
            "class_labels": list(range(NUM_CLASSES)),
            "model_available": self.model_loaded,
            "total_params": None,
            "layers": None,
            "framework": None,
            "evaluation": None,
        }
        if self._model is not None:
            summary = model_summary_dict(self._model)
            info.update(
                {
                    "name": summary["name"],
                    "total_params": summary["total_params"],
                    "layers": summary["layers"],
                    "framework": summary["framework"],
                }
            )
        return info


_predictor: PredictorService | None = None


def get_predictor() -> PredictorService:
    global _predictor
    if _predictor is None:
        _predictor = PredictorService()
        _predictor.load()
    return _predictor
