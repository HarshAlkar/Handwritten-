"""REST API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, status

from app.ml.datasets import dataset_summary, get_dataset, list_dataset_names
from app.ml.persistence import load_metadata
from app.schemas.api import CompareRequest, HealthResponse, PredictRequest, TrainRequest
from app.services.training import compare_algorithms, get_models, predict_with_model, train_model
from app.utils.json_safe import json_safe

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/datasets")
def datasets():
    items = []
    for name in list_dataset_names():
        items.append(dataset_summary(get_dataset(name)))
    return {"datasets": items}


@router.get("/datasets/{dataset_name}/sample")
def dataset_sample(dataset_name: str, limit: int = Query(default=8, ge=1, le=50)):
    try:
        bundle = get_dataset(dataset_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    df = bundle.to_dataframe().head(limit)
    return {
        "dataset": bundle.name,
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }


@router.post("/train")
def train(req: TrainRequest):
    try:
        return train_model(
            dataset_name=req.dataset,
            algorithm=req.algorithm,
            hyperparameters=req.hyperparameters,
            test_size=req.test_size,
            random_seed=req.random_seed,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Training failed")
        raise HTTPException(
            status_code=500,
            detail="Training failed due to an internal error.",
        ) from None


@router.get("/models")
def models():
    return json_safe({"models": get_models()})


@router.post("/predict")
def predict(req: PredictRequest):
    try:
        return predict_with_model(req.model_id, req.features)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from None


@router.get("/models/{model_id}/metrics")
def model_metrics(model_id: str):
    try:
        meta = load_metadata(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "model_id": model_id,
        "algorithm": meta.get("algorithm"),
        "dataset": meta.get("dataset"),
        "train_seconds": meta.get("train_seconds"),
        "metrics": meta.get("metrics"),
        "hyperparameters": meta.get("hyperparameters"),
        "adaboost_diagnostics": meta.get("adaboost_diagnostics"),
    }


@router.post("/compare")
def compare(req: CompareRequest):
    try:
        return compare_algorithms(
            dataset_name=req.dataset,
            rf_params=req.random_forest,
            ada_params=req.adaboost,
            test_size=req.test_size,
            random_seed=req.random_seed,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Comparison failed")
        raise HTTPException(status_code=500, detail="Comparison failed.") from None


@router.get("/models/{model_id}/feature-importance")
def feature_importance(model_id: str, top_k: int = Query(default=15, ge=1, le=50)):
    try:
        meta = load_metadata(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    importance = meta.get("feature_importance")
    if not importance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feature importance is not available for this model.",
        )
    return {
        "model_id": model_id,
        "algorithm": meta.get("algorithm"),
        "features": importance[:top_k],
        "disclaimer": (
            "Tree-based feature importance reflects impurity reduction on the training data "
            "and does not prove causation."
        ),
    }
