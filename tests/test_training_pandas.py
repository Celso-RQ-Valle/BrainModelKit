"""Training behavior and artifact tests."""

import json

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from brainmodelkit.persistence import load_model
from brainmodelkit.training.pandas import train_model


@pytest.mark.parametrize(
    "model",
    ["logistic_regression", "random_forest", "gradient_boosting", LogisticRegression()],
)
def test_train_score_and_report(tmp_path, model, monkeypatch):
    def unexpected_save(*args, **kwargs):
        pytest.fail("Training without save_path must not persist the model")

    monkeypatch.setattr(
        "brainmodelkit.training.pandas._save_python_model", unexpected_save
    )
    data = pd.DataFrame(
        {"x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], "y": [0, 0, 0, 1, 1, 1]}
    )
    original = data.copy()
    result = train_model(
        "test",
        "y",
        ["x"],
        data,
        data,
        model=model,
        df_scoring=data[["x"]],
        output_dir=tmp_path,
    )
    assert result.metrics["oot_auc"] == 1
    assert result.metrics["oot_ks"] == 1
    assert result.scoring_predictions["score"].between(0, 1).all()
    pd.testing.assert_frame_equal(data, original)
    report = json.loads((result.output_dir / "model_info.json").read_text())
    assert report["feature_cols"] == ["x"]
    assert len(result.feature_importance) == 1
    if not isinstance(model, str):
        assert not hasattr(model, "classes_")


def test_invalid_and_empty_data(tmp_path):
    data = pd.DataFrame({"x": [-1.0, 1.0], "y": [0, 1]})
    with pytest.raises(ValueError, match="must not include"):
        train_model("bad", "y", ["y"], data, data, output_dir=tmp_path)
    with pytest.raises(ValueError, match="both target classes"):
        train_model("bad", "y", ["x"], data.iloc[:1], data, output_dir=tmp_path)
    result = train_model("empty", "y", ["x"], data, data.iloc[:0], output_dir=tmp_path)
    assert result.oot_predictions.empty
    report = json.loads((result.output_dir / "model_info.json").read_text())
    assert report["metrics"]["oot_auc"] is None


def test_mlflow_round_trip(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    flavor = pytest.importorskip("mlflow.sklearn")

    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    try:
        mlflow.set_experiment("training-test")
        data = pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0], "y": [0, 0, 1, 1]})
        result = train_model(
            "logged",
            "y",
            ["x"],
            data,
            data,
            output_dir=tmp_path / "reports",
            mlflow_logging=True,
            signature=True,
        )
        loaded = flavor.load_model(f"runs:/{result.run_id}/model")
        assert list(loaded.predict(data[["x"]])) == list(data.y)
        assert (result.output_dir / "mlflow_run_id.txt").exists()
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_lightgbm(tmp_path):
    pytest.importorskip("lightgbm")
    data = pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0] * 10, "y": [0, 0, 1, 1] * 10})
    result = train_model(
        "lgbm",
        "y",
        ["x"],
        data,
        data,
        model="lightgbm",
        model_params={"min_child_samples": 1, "verbosity": -1},
        output_dir=tmp_path,
    )
    assert result.metrics["oot_auc"] == 1


def test_training_persistence(tmp_path):
    data = pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0], "y": [0, 0, 1, 1]})
    path = tmp_path / "model.pkl"
    result = train_model(
        "saved", "y", ["x"], data, data, output_dir=tmp_path / "reports", save_path=path
    )
    loaded = load_model(path)
    assert list(loaded.predict(data[["x"]])) == list(result.model.predict(data[["x"]]))
    assert (
        json.loads((tmp_path / "model.pkl.metadata.json").read_text())["target_col"]
        == "y"
    )
    with pytest.raises(ValueError, match="Available:"):
        train_model("bad", "y", ["x"], data, data, save_format="spark")
