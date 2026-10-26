from fastapi.testclient import TestClient

from app.main import app
from app.ml.datasets import get_dataset


client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_datasets():
    res = client.get("/api/datasets")
    assert res.status_code == 200
    names = {d["name"] for d in res.json()["datasets"]}
    assert "breast_cancer" in names
    assert "iris" in names


def test_sample():
    res = client.get("/api/datasets/breast_cancer/sample?limit=3")
    assert res.status_code == 200
    assert len(res.json()["rows"]) == 3


def test_train_predict_flow():
    train = client.post(
        "/api/train",
        json={
            "dataset": "breast_cancer",
            "algorithm": "random_forest",
            "hyperparameters": {"n_estimators": 25},
            "random_seed": 11,
        },
    )
    assert train.status_code == 200
    body = train.json()
    model_id = body["model_id"]
    assert "metrics" in body

    bundle = get_dataset("breast_cancer")
    features = {n: float(bundle.X[5, i]) for i, n in enumerate(bundle.feature_names)}
    pred = client.post("/api/predict", json={"model_id": model_id, "features": features})
    assert pred.status_code == 200
    assert pred.json()["predicted_class"] in bundle.target_names


def test_unknown_model():
    res = client.post("/api/predict", json={"model_id": "does_not_exist", "features": {"a": 1}})
    assert res.status_code in (400, 404)


def test_compare():
    res = client.post(
        "/api/compare",
        json={
            "dataset": "breast_cancer",
            "random_forest": {"n_estimators": 20},
            "adaboost": {"n_estimators": 20},
            "random_seed": 5,
        },
    )
    assert res.status_code == 200
    models = res.json()["models"]
    assert "accuracy" in models["random_forest"]["metrics"]
    assert "accuracy" in models["adaboost"]["metrics"]
