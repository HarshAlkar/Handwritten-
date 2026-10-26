"""Evaluation metrics for trained classifiers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str],
    task_type: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_seconds = time.perf_counter() - t0

    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)

    if task_type == "binary":
        precision = float(precision_score(y_test, y_pred, average="binary", zero_division=0, pos_label=1))
        recall = float(recall_score(y_test, y_pred, average="binary", zero_division=0, pos_label=1))
        f1 = float(f1_score(y_test, y_pred, average="binary", zero_division=0, pos_label=1))
        average_label = "binary (positive class index=1)"
        positive_class = target_names[1] if len(target_names) > 1 else None
    else:
        precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        average_label = "macro"
        positive_class = None

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average": average_label,
        "positive_class": positive_class,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "predict_seconds": float(predict_seconds),
        "n_test_samples": int(len(y_test)),
    }

    roc_auc = None
    roc = None
    try:
        if task_type == "binary" and probs is not None and probs.shape[1] == 2:
            scores = probs[:, 1]
            roc_auc = float(roc_auc_score(y_test, scores))
            fpr, tpr, thresholds = roc_curve(y_test, scores)
            step = max(1, len(fpr) // 100)
            thr = []
            for t in thresholds[::step]:
                t = float(t)
                thr.append(None if np.isnan(t) or np.isinf(t) else t)
            roc = {
                "fpr": [float(x) for x in fpr[::step]],
                "tpr": [float(x) for x in tpr[::step]],
                "thresholds": thr,
            }
        elif task_type == "multiclass" and probs is not None:
            roc_auc = float(roc_auc_score(y_test, probs, multi_class="ovr", average="macro"))
    except ValueError:
        roc_auc = None
        roc = None

    metrics["roc_auc"] = roc_auc
    metrics["roc_curve"] = roc
    return metrics


def feature_importance_payload(model: Any, feature_names: list[str]) -> list[dict[str, Any]] | None:
    if not hasattr(model, "feature_importances_"):
        return None
    importances = np.asarray(model.feature_importances_, dtype=float)
    ranked = sorted(
        [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in range(len(feature_names))
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )
    return ranked


def adaboost_diagnostics(model: Any) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    if hasattr(model, "estimator_weights_"):
        payload["estimator_weights"] = [float(x) for x in model.estimator_weights_]
    if hasattr(model, "estimator_errors_"):
        payload["estimator_errors"] = [float(x) for x in model.estimator_errors_]
    return payload or None
