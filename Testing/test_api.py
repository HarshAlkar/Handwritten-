"""API integration tests for FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from Backend.app.main import app
from Backend.app.services import predictor as predictor_mod

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "ML" / "artifacts" / "digit_cnn.keras"


@pytest.fixture
def client():
    # Reset singleton so each session reloads model if present
    predictor_mod._predictor = None
    with TestClient(app) as c:
        yield c
    predictor_mod._predictor = None


def test_health(client: TestClient):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert "status" in body
    assert "model_loaded" in body
    assert isinstance(body["model_loaded"], bool)


def test_predict_requires_image(client: TestClient):
    if not MODEL_PATH.exists():
        pytest.skip("Trained model required")
    res = client.post("/api/predict")
    assert res.status_code == 400


def test_predict_empty_image(client: TestClient, empty_canvas_png: bytes):
    if not MODEL_PATH.exists():
        pytest.skip("Trained model required")
    res = client.post(
        "/api/predict",
        files={"file": ("empty.png", empty_canvas_png, "image/png")},
    )
    assert res.status_code == 400
    assert "detail" in res.json()


def test_predict_valid_digit(client: TestClient, white_bg_digit_png: bytes):
    if not MODEL_PATH.exists():
        pytest.skip("Trained model required")
    res = client.post(
        "/api/predict",
        files={"file": ("digit.png", white_bg_digit_png, "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["predicted_digit"] <= 9
    assert 0.0 <= body["confidence"] <= 1.0
    probs = body["probabilities"]
    assert len(probs) == 10
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-4


def test_model_info(client: TestClient):
    res = client.get("/api/model-info")
    assert res.status_code == 200
    body = res.json()
    assert body["input_shape"] == [28, 28, 1]
    assert body["num_classes"] == 10
    assert body["class_labels"] == list(range(10))


def test_metrics(client: TestClient):
    res = client.get("/api/metrics")
    assert res.status_code == 200
    body = res.json()
    assert "available" in body
