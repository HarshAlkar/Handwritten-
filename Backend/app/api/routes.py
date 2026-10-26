"""REST API routes for digit recognition."""

from __future__ import annotations

import base64
import binascii
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from Backend.app.config import get_settings
from Backend.app.schemas.predict import (
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictResponse,
)
from Backend.app.services.metrics_loader import (
    load_history,
    load_metrics,
    public_evaluation,
)
from Backend.app.services.predictor import get_predictor
from ML.preprocessing.pipeline import PreprocessError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    predictor = get_predictor()
    return HealthResponse(
        status="ok" if predictor.model_loaded else "degraded",
        model_loaded=predictor.model_loaded,
        model_path_configured=settings.resolved_model_path().exists(),
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile | None = File(default=None),
    image_base64: str | None = Form(default=None),
) -> PredictResponse:
    settings = get_settings()
    predictor = get_predictor()

    if not predictor.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available. Train the model and restart the server.",
        )

    data: bytes | None = None
    content_type: str | None = None
    filename: str | None = None

    if file is not None and file.filename:
        content_type = file.content_type
        filename = file.filename
        data = await file.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Uploaded file exceeds the maximum allowed size.",
            )
    elif image_base64:
        payload = image_base64.strip()
        if "," in payload and payload.lower().startswith("data:"):
            header, payload = payload.split(",", 1)
            if "image/" in header:
                content_type = header.split(";")[0].replace("data:", "")
        try:
            data = base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid base64 image data.",
            ) from exc
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Uploaded image exceeds the maximum allowed size.",
            )
        filename = "canvas.png"
        content_type = content_type or "image/png"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide an image file or image_base64 field.",
        )

    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image payload.",
        )

    try:
        result = predictor.predict_from_bytes(
            data, content_type=content_type, filename=filename
        )
    except PreprocessError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal error.",
        ) from None

    return PredictResponse(**result)


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    predictor = get_predictor()
    if not predictor.model_loaded:
        predictor.load()
    info = predictor.model_info()
    metrics = public_evaluation(load_metrics())
    info["evaluation"] = metrics
    return ModelInfoResponse(**info)


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    history = load_history()
    evaluation = public_evaluation(load_metrics())
    available = history is not None or evaluation is not None
    message = None
    if not available:
        message = "No metrics available yet. Train and evaluate the model first."
    return MetricsResponse(
        history=history,
        evaluation=evaluation,
        available=available,
        message=message,
    )
