import numpy as np
import pytest

from app.ml.datasets import get_dataset
from app.ml.models import build_adaboost, build_random_forest, validate_rf_params
from app.ml.preprocessing import prepare_split
from app.services.training import compare_algorithms, predict_with_model, train_model


def test_invalid_hyperparams():
    with pytest.raises(ValueError):
        validate_rf_params({"n_estimators": 0})


def test_train_rf_and_predict():
    result = train_model("breast_cancer", "random_forest", {"n_estimators": 30}, test_size=0.25, random_seed=7)
    assert result["metrics"]["accuracy"] > 0.85
    assert result["feature_importance"]
    model_id = result["model_id"]

    bundle = get_dataset("breast_cancer")
    sample = {name: float(bundle.X[0, i]) for i, name in enumerate(bundle.feature_names)}
    pred = predict_with_model(model_id, sample)
    assert pred["predicted_class"] in bundle.target_names
    assert pred["probabilities"] is not None
    assert abs(sum(pred["probabilities"].values()) - 1.0) < 1e-5


def test_train_adaboost():
    result = train_model("iris", "adaboost", {"n_estimators": 40, "learning_rate": 0.8}, random_seed=3)
    assert result["metrics"]["accuracy"] > 0.8
    assert result["task_type"] == "multiclass"
    probs_ok = result["metrics"]["accuracy"] <= 1.0


def test_compare_same_split():
    cmp = compare_algorithms(
        "breast_cancer",
        rf_params={"n_estimators": 40},
        ada_params={"n_estimators": 40},
        random_seed=42,
    )
    assert "random_forest" in cmp["models"]
    assert "adaboost" in cmp["models"]
    assert cmp["models"]["random_forest"]["n_train"] == cmp["models"]["adaboost"]["n_train"]
    assert cmp["models"]["random_forest"]["n_test"] == cmp["models"]["adaboost"]["n_test"]


def test_probability_shape_rf():
    bundle = get_dataset("breast_cancer")
    split = prepare_split(bundle, random_seed=1)
    model = build_random_forest({"n_estimators": 20, "random_state": 1})
    model.fit(split.X_train, split.y_train)
    probs = model.predict_proba(split.X_test)
    assert probs.shape[1] == 2
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(probs)), atol=1e-5)


def test_adaboost_builds():
    model = build_adaboost({"n_estimators": 10, "learning_rate": 1.0, "random_state": 0})
    assert model.n_estimators == 10
