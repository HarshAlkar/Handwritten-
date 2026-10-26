"""Dataset loading and metadata for supervised learning experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    display_name: str
    task_type: str  # binary | multiclass
    feature_names: list[str]
    target_names: list[str]
    X: np.ndarray
    y: np.ndarray
    description: str

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_classes(self) -> int:
        return len(self.target_names)

    def class_distribution(self) -> dict[str, int]:
        unique, counts = np.unique(self.y, return_counts=True)
        return {self.target_names[int(u)]: int(c) for u, c in zip(unique, counts)}

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df["target"] = [self.target_names[int(v)] for v in self.y]
        return df

    def feature_ranges(self) -> dict[str, dict[str, float]]:
        ranges: dict[str, dict[str, float]] = {}
        for i, name in enumerate(self.feature_names):
            col = self.X[:, i]
            ranges[name] = {
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
            }
        return ranges


def _validate_bundle(bundle: DatasetBundle) -> None:
    if bundle.X.ndim != 2:
        raise ValueError("Feature matrix must be 2-dimensional.")
    if bundle.y.ndim != 1:
        raise ValueError("Target vector must be 1-dimensional.")
    if len(bundle.X) != len(bundle.y):
        raise ValueError("X and y length mismatch.")
    if np.isnan(bundle.X).any() or np.isnan(bundle.y.astype(float)).any():
        raise ValueError("Dataset contains missing values.")
    if len(bundle.feature_names) != bundle.X.shape[1]:
        raise ValueError("Feature name count mismatch.")


def load_breast_cancer_dataset() -> DatasetBundle:
    raw = load_breast_cancer()
    bundle = DatasetBundle(
        name="breast_cancer",
        display_name="Breast Cancer Wisconsin",
        task_type="binary",
        feature_names=list(raw.feature_names),
        target_names=[str(n) for n in raw.target_names],
        X=np.asarray(raw.data, dtype=np.float64),
        y=np.asarray(raw.target, dtype=np.int64),
        description=(
            "Binary classification of breast mass biopsies as malignant or benign "
            "using 30 numeric features computed from digitized images."
        ),
    )
    _validate_bundle(bundle)
    return bundle


def load_iris_dataset() -> DatasetBundle:
    raw = load_iris()
    bundle = DatasetBundle(
        name="iris",
        display_name="Iris Plants",
        task_type="multiclass",
        feature_names=list(raw.feature_names),
        target_names=[str(n) for n in raw.target_names],
        X=np.asarray(raw.data, dtype=np.float64),
        y=np.asarray(raw.target, dtype=np.int64),
        description=(
            "Multiclass classification of Iris flowers into setosa, versicolor, "
            "and virginica using sepal and petal measurements."
        ),
    )
    _validate_bundle(bundle)
    return bundle


DATASETS: dict[str, Callable[[], DatasetBundle]] = {
    "breast_cancer": load_breast_cancer_dataset,
    "iris": load_iris_dataset,
}


def list_dataset_names() -> list[str]:
    return list(DATASETS.keys())


def get_dataset(name: str) -> DatasetBundle:
    key = name.strip().lower()
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(DATASETS)}")
    return DATASETS[key]()


def dataset_summary(bundle: DatasetBundle) -> dict[str, Any]:
    return {
        "name": bundle.name,
        "display_name": bundle.display_name,
        "task_type": bundle.task_type,
        "n_samples": bundle.n_samples,
        "n_features": bundle.n_features,
        "n_classes": bundle.n_classes,
        "feature_names": bundle.feature_names,
        "target_names": bundle.target_names,
        "class_distribution": bundle.class_distribution(),
        "feature_ranges": bundle.feature_ranges(),
        "description": bundle.description,
        "positive_class": bundle.target_names[1] if bundle.task_type == "binary" else None,
    }
