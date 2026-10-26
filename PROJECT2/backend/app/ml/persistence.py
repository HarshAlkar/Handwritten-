"""Joblib model persistence and metadata tracking."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn

from app.config import get_settings
from app.utils.json_safe import json_safe


def new_model_id(algorithm: str, dataset: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{algorithm}_{dataset}_{stamp}_{short}"


def save_experiment(
    model: Any,
    metadata: dict[str, Any],
) -> str:
    settings = get_settings()
    model_id = metadata.get("model_id") or new_model_id(
        metadata.get("algorithm", "model"),
        metadata.get("dataset", "data"),
    )
    metadata["model_id"] = model_id
    metadata["sklearn_version"] = sklearn.__version__
    metadata["saved_at"] = datetime.now(timezone.utc).isoformat()

    model_path = settings.models_dir() / f"{model_id}.joblib"
    meta_path = settings.metadata_dir() / f"{model_id}.json"

    # Only save models we created ourselves into our artifacts directory
    joblib.dump(
        {
            "model": model,
            "scaler": metadata.get("_scaler_object"),
            "feature_names": metadata.get("feature_names"),
            "target_names": metadata.get("target_names"),
        },
        model_path,
    )

    public_meta = json_safe({k: v for k, v in metadata.items() if not k.startswith("_")})
    public_meta["artifact_file"] = model_path.name
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(public_meta, f, indent=2, allow_nan=False)

    return model_id


def list_models() -> list[dict[str, Any]]:
    settings = get_settings()
    rows: list[dict[str, Any]] = []
    for path in sorted(settings.metadata_dir().glob("*.json"), reverse=True):
        try:
            with open(path, encoding="utf-8") as f:
                rows.append(json_safe(json.load(f)))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def load_metadata(model_id: str) -> dict[str, Any]:
    settings = get_settings()
    path = settings.metadata_dir() / f"{model_id}.json"
    if not path.exists():
        raise KeyError("Unknown model_id.")
    with open(path, encoding="utf-8") as f:
        return json_safe(json.load(f))


def load_artifact(model_id: str) -> dict[str, Any]:
    """Load a model artifact that was saved by this application only."""
    settings = get_settings()
    # Resolve only within artifacts/models to avoid path traversal / untrusted files
    safe_name = Path(model_id).name
    if safe_name != model_id or ".." in model_id:
        raise ValueError("Invalid model_id.")
    model_path = settings.models_dir() / f"{safe_name}.joblib"
    if not model_path.exists():
        raise KeyError("Unknown model_id.")
    # Trusted: only files written by this app into artifacts/models
    return joblib.load(model_path)
