"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AlgorithmName = Literal["random_forest", "adaboost"]
DatasetName = Literal["breast_cancer", "iris"]


class HealthResponse(BaseModel):
    status: str
    service: str = "supervised-learning-explorer"


class TrainRequest(BaseModel):
    dataset: DatasetName = "breast_cancer"
    algorithm: AlgorithmName
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_seed: int = Field(default=42, ge=0)


class CompareRequest(BaseModel):
    dataset: DatasetName = "breast_cancer"
    random_forest: dict[str, Any] = Field(default_factory=dict)
    adaboost: dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_seed: int = Field(default=42, ge=0)


class PredictRequest(BaseModel):
    model_id: str
    features: dict[str, float]


class ErrorResponse(BaseModel):
    detail: str
