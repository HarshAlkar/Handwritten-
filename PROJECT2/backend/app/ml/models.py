"""Random Forest and AdaBoost model builders (scikit-learn)."""

from __future__ import annotations

from typing import Any, Literal

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

AlgorithmName = Literal["random_forest", "adaboost"]


def validate_rf_params(params: dict[str, Any], max_n_estimators: int = 500) -> dict[str, Any]:
    n_estimators = int(params.get("n_estimators", 100))
    max_depth = params.get("max_depth", None)
    min_samples_split = int(params.get("min_samples_split", 2))
    min_samples_leaf = int(params.get("min_samples_leaf", 1))
    max_features = params.get("max_features", "sqrt")
    random_state = int(params.get("random_state", 42))

    if not 1 <= n_estimators <= max_n_estimators:
        raise ValueError(f"n_estimators must be between 1 and {max_n_estimators}.")
    if max_depth is not None:
        max_depth = int(max_depth)
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1 or null.")
    if min_samples_split < 2:
        raise ValueError("min_samples_split must be >= 2.")
    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be >= 1.")
    if max_features not in {"sqrt", "log2", None} and not isinstance(max_features, (int, float)):
        raise ValueError("max_features must be 'sqrt', 'log2', null, or a number.")

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "random_state": random_state,
    }


def validate_adaboost_params(params: dict[str, Any], max_n_estimators: int = 500) -> dict[str, Any]:
    n_estimators = int(params.get("n_estimators", 50))
    learning_rate = float(params.get("learning_rate", 1.0))
    random_state = int(params.get("random_state", 42))
    max_depth = int(params.get("base_max_depth", 1))

    if not 1 <= n_estimators <= max_n_estimators:
        raise ValueError(f"n_estimators must be between 1 and {max_n_estimators}.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if max_depth < 1:
        raise ValueError("base_max_depth must be >= 1.")

    return {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "random_state": random_state,
        "base_max_depth": max_depth,
    }


def build_random_forest(params: dict[str, Any], max_n_estimators: int = 500) -> RandomForestClassifier:
    cfg = validate_rf_params(params, max_n_estimators=max_n_estimators)
    return RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_split"],
        min_samples_leaf=cfg["min_samples_leaf"],
        max_features=cfg["max_features"],
        random_state=cfg["random_state"],
        n_jobs=-1,
    )


def build_adaboost(params: dict[str, Any], max_n_estimators: int = 500) -> AdaBoostClassifier:
    """Build AdaBoost with shallow decision trees as weak learners (sklearn >= 1.2 uses `estimator`)."""
    cfg = validate_adaboost_params(params, max_n_estimators=max_n_estimators)
    weak = DecisionTreeClassifier(
        max_depth=cfg["base_max_depth"],
        random_state=cfg["random_state"],
    )
    # sklearn >= 1.6: only SAMME-style boosting; `algorithm` kwarg removed
    return AdaBoostClassifier(
        estimator=weak,
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        random_state=cfg["random_state"],
    )


def build_model(
    algorithm: AlgorithmName,
    params: dict[str, Any],
    max_n_estimators: int = 500,
):
    if algorithm == "random_forest":
        return build_random_forest(params, max_n_estimators=max_n_estimators)
    if algorithm == "adaboost":
        return build_adaboost(params, max_n_estimators=max_n_estimators)
    raise ValueError(f"Unsupported algorithm: {algorithm}")
