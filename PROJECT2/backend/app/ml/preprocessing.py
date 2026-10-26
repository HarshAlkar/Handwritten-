"""Train/test splitting and scaling without data leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .datasets import DatasetBundle


@dataclass
class PreparedSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    test_size: float
    random_seed: int

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)


def prepare_split(
    bundle: DatasetBundle,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> PreparedSplit:
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5.")

    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=test_size,
        random_state=random_seed,
        stratify=bundle.y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return PreparedSplit(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        test_size=test_size,
        random_seed=random_seed,
    )
