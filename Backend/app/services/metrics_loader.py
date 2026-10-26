"""Load training history and evaluation metrics from artifact JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Backend.app.config import Settings, get_settings


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def load_history(settings: Settings | None = None) -> dict[str, Any] | None:
    settings = settings or get_settings()
    return _read_json(settings.resolved_history_path())


def load_metrics(settings: Settings | None = None) -> dict[str, Any] | None:
    settings = settings or get_settings()
    return _read_json(settings.resolved_metrics_path())


def public_evaluation(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip internal paths from evaluation payload for API responses."""
    if not metrics:
        return None
    keys = [
        "dataset",
        "test_samples",
        "test_loss",
        "test_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "confusion_matrix",
        "per_class",
        "macro_avg",
        "weighted_avg",
    ]
    return {k: metrics[k] for k in keys if k in metrics}
