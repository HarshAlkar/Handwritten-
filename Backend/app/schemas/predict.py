"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    predicted_digit: int = Field(..., ge=0, le=9)
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path_configured: bool


class ModelInfoResponse(BaseModel):
    name: str
    input_shape: list[int]
    num_classes: int
    class_labels: list[int]
    total_params: int | None = None
    layers: list[dict[str, Any]] | None = None
    framework: str | None = None
    evaluation: dict[str, Any] | None = None
    model_available: bool


class MetricsResponse(BaseModel):
    history: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    available: bool
    message: str | None = None


class ErrorResponse(BaseModel):
    detail: str
