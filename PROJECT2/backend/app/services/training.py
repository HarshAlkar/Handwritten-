"""Training, prediction, and comparison services."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from app.config import get_settings
from app.ml.datasets import get_dataset
from app.ml.metrics import adaboost_diagnostics, evaluate_classifier, feature_importance_payload
from app.ml.models import AlgorithmName, build_model, validate_adaboost_params, validate_rf_params
from app.ml.persistence import list_models, load_artifact, load_metadata, new_model_id, save_experiment
from app.ml.preprocessing import prepare_split
from app.utils.json_safe import json_safe


def _validated_params(algorithm: AlgorithmName, params: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    if algorithm == "random_forest":
        return validate_rf_params(params, max_n_estimators=settings.max_n_estimators)
    return validate_adaboost_params(params, max_n_estimators=settings.max_n_estimators)


def train_model(
    dataset_name: str,
    algorithm: AlgorithmName,
    hyperparameters: dict[str, Any] | None = None,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict[str, Any]:
    hyperparameters = hyperparameters or {}
    bundle = get_dataset(dataset_name)
    params = _validated_params(algorithm, {**hyperparameters, "random_state": hyperparameters.get("random_state", random_seed)})
    split = prepare_split(bundle, test_size=test_size, random_seed=random_seed)
    model = build_model(algorithm, params, max_n_estimators=get_settings().max_n_estimators)

    t0 = time.perf_counter()
    model.fit(split.X_train, split.y_train)
    train_seconds = time.perf_counter() - t0

    metrics = evaluate_classifier(
        model,
        split.X_test,
        split.y_test,
        bundle.target_names,
        bundle.task_type,
    )
    importance = feature_importance_payload(model, bundle.feature_names)
    boost_diag = adaboost_diagnostics(model) if algorithm == "adaboost" else None

    model_id = new_model_id(algorithm, bundle.name)
    metadata = {
        "model_id": model_id,
        "algorithm": algorithm,
        "dataset": bundle.name,
        "dataset_display_name": bundle.display_name,
        "task_type": bundle.task_type,
        "hyperparameters": params,
        "test_size": split.test_size,
        "random_seed": split.random_seed,
        "n_train": int(len(split.y_train)),
        "n_test": int(len(split.y_test)),
        "feature_names": bundle.feature_names,
        "target_names": bundle.target_names,
        "train_seconds": float(train_seconds),
        "metrics": metrics,
        "feature_importance": importance,
        "adaboost_diagnostics": boost_diag,
        "_scaler_object": split.scaler,
    }
    save_experiment(model, metadata)
    return json_safe({k: v for k, v in metadata.items() if not k.startswith("_")})


def compare_algorithms(
    dataset_name: str,
    rf_params: dict[str, Any] | None = None,
    ada_params: dict[str, Any] | None = None,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train both algorithms on the exact same stratified split."""
    rf_params = rf_params or {}
    ada_params = ada_params or {}
    bundle = get_dataset(dataset_name)
    split = prepare_split(bundle, test_size=test_size, random_seed=random_seed)

    results: dict[str, Any] = {
        "dataset": bundle.name,
        "task_type": bundle.task_type,
        "test_size": test_size,
        "random_seed": random_seed,
        "n_train": int(len(split.y_train)),
        "n_test": int(len(split.y_test)),
        "target_names": bundle.target_names,
        "models": {},
    }

    for algorithm, params in (("random_forest", rf_params), ("adaboost", ada_params)):
        validated = _validated_params(algorithm, {**params, "random_state": params.get("random_state", random_seed)})
        model = build_model(algorithm, validated, max_n_estimators=get_settings().max_n_estimators)
        t0 = time.perf_counter()
        model.fit(split.X_train, split.y_train)
        train_seconds = time.perf_counter() - t0
        metrics = evaluate_classifier(
            model, split.X_test, split.y_test, bundle.target_names, bundle.task_type
        )
        model_id = new_model_id(algorithm, bundle.name)
        metadata = {
            "model_id": model_id,
            "algorithm": algorithm,
            "dataset": bundle.name,
            "dataset_display_name": bundle.display_name,
            "task_type": bundle.task_type,
            "hyperparameters": validated,
            "test_size": test_size,
            "random_seed": random_seed,
            "n_train": int(len(split.y_train)),
            "n_test": int(len(split.y_test)),
            "feature_names": bundle.feature_names,
            "target_names": bundle.target_names,
            "train_seconds": float(train_seconds),
            "metrics": metrics,
            "feature_importance": feature_importance_payload(model, bundle.feature_names),
            "adaboost_diagnostics": adaboost_diagnostics(model) if algorithm == "adaboost" else None,
            "_scaler_object": split.scaler,
        }
        save_experiment(model, metadata)
        results["models"][algorithm] = {k: v for k, v in metadata.items() if not k.startswith("_")}

    return json_safe(results)


def predict_with_model(model_id: str, features: dict[str, float]) -> dict[str, Any]:
    meta = load_metadata(model_id)
    artifact = load_artifact(model_id)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names: list[str] = artifact["feature_names"]
    target_names: list[str] = artifact["target_names"]

    missing = [f for f in feature_names if f not in features]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}")
    extra = [f for f in features if f not in feature_names]
    if extra:
        raise ValueError(f"Unknown features: {', '.join(extra[:5])}")

    row = np.asarray([[float(features[name]) for name in feature_names]], dtype=np.float64)
    if np.isnan(row).any():
        raise ValueError("Feature values must be finite numbers.")
    scaled = scaler.transform(row)
    pred_idx = int(model.predict(scaled)[0])
    if pred_idx < 0 or pred_idx >= len(target_names):
        raise RuntimeError("Model returned an unexpected class index.")

    probabilities = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled)[0]
        probabilities = {target_names[i]: float(probs[i]) for i in range(len(target_names))}
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}

    return {
        "model_id": model_id,
        "dataset": meta.get("dataset"),
        "algorithm": meta.get("algorithm"),
        "predicted_class": target_names[pred_idx],
        "predicted_index": pred_idx,
        "probabilities": probabilities,
        "confidence": max(probabilities.values()) if probabilities else None,
    }


def get_models() -> list[dict[str, Any]]:
    return list_models()
