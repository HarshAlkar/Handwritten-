from app.ml.datasets import get_dataset, list_dataset_names
from app.ml.preprocessing import prepare_split


def test_list_datasets():
    names = list_dataset_names()
    assert "breast_cancer" in names
    assert "iris" in names


def test_breast_cancer_shape():
    b = get_dataset("breast_cancer")
    assert b.n_samples == 569
    assert b.n_features == 30
    assert b.task_type == "binary"
    assert set(b.class_distribution().keys()) == {"malignant", "benign"}


def test_iris_multiclass():
    b = get_dataset("iris")
    assert b.n_classes == 3
    assert b.task_type == "multiclass"


def test_stratified_split_no_leakage():
    b = get_dataset("breast_cancer")
    split = prepare_split(b, test_size=0.2, random_seed=42)
    assert len(split.X_train) + len(split.X_test) == b.n_samples
    # Scaler fitted on train only: mean of scaled train ~0
    assert abs(float(split.X_train.mean())) < 0.05
