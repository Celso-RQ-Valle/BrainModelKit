"""RFE elimination, validation and model integration tests."""

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier

from brainmodelkit.feature_selection._common import run_rfe
from brainmodelkit.feature_selection.pandas import rfe
from brainmodelkit.training import TrainingResult


def test_elimination_order_and_exact_final_count(tmp_path):
    data = pd.DataFrame(columns=["a", "b", "c", "d", "target"])
    calls = []

    def trainer(**kwargs):
        features = kwargs["feature_cols"]
        calls.append(features)
        folder = Path(kwargs["output_dir"]) / str(len(features))
        folder.mkdir(parents=True)
        values = {"a": -10.0, "b": 0.0, "c": 0.0, "d": 1.0}
        return TrainingResult(
            None,
            None,
            None,
            {"oot_ks": float("nan")},
            [{"feature": f, "importance": values[f]} for f in features],
            folder,
        )

    result = run_rfe(
        trainer,
        "test",
        "target",
        ["a", "b", "c", "d"],
        data,
        data,
        "custom",
        2,
        1,
        None,
        tmp_path,
        None,
        False,
        False,
    )
    assert calls == [["a", "b", "c", "d"], ["a", "d"], ["a"]]
    assert result.selected_features == ["a"]
    assert result.history[0]["removed_features"] == ["b", "c"]
    assert result.history[-1]["removed_features"] == []
    assert json.loads((result.output_dir / "selected_features.json").read_text()) == [
        "a"
    ]
    assert json.loads((result.output_dir / "history.json").read_text())[0][
        "metrics"
    ] == {"oot_ks": None}


@pytest.mark.parametrize(
    "model", ["logistic_regression", "random_forest", "gradient_boosting", "lightgbm"]
)
def test_pandas_models(tmp_path, model):
    if model == "lightgbm":
        pytest.importorskip("lightgbm")
    X, y = make_classification(n_samples=80, n_features=4, random_state=42)
    features = [f"feat_{i}" for i in range(4)]
    data = pd.DataFrame(X, columns=features).assign(target=y)
    original = data.copy()
    params = {"random_state": 42}
    if model == "lightgbm":
        params.update(verbosity=-1, min_child_samples=2, n_estimators=5)
    result = rfe(
        "test",
        "target",
        data,
        data,
        model=model,
        model_params=params,
        step=3,
        n_feat_final=2,
        output_dir=tmp_path,
        df_scoring=data.drop(columns="target"),
    )
    assert len(result.selected_features) == 2
    assert [h["n_features"] for h in result.history] == [4, 2]
    assert result.training_result.scoring_predictions["score"].between(0, 1).all()
    assert result.training_result.model.n_features_in_ == 2
    pd.testing.assert_frame_equal(original, data)
    for iteration in result.history:
        assert (Path(iteration["report_dir"]) / "feature_importance.csv").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"step": 0},
        {"step": True},
        {"step": 1.5},
        {"n_feat_final": 0},
        {"n_feat_final": 3},
        {"feature_cols": []},
        {"feature_cols": ["target"]},
    ],
)
def test_invalid_options(tmp_path, options):
    data = pd.DataFrame({"feat_a": [-1.0, 1.0], "feat_b": [0.0, 0.0], "target": [0, 1]})
    with pytest.raises(ValueError):
        rfe("test", "target", data, data, output_dir=tmp_path, **options)
    assert list(tmp_path.iterdir()) == []


def test_equal_count_trains_once_and_unsupported_model_fails(tmp_path):
    data = pd.DataFrame({"feat_a": [-1.0, 1.0], "target": [0, 1]})
    result = rfe("test", "target", data, data, n_feat_final=1, output_dir=tmp_path)
    assert len(result.history) == 1
    with pytest.raises(ValueError, match="native importance"):
        rfe("test", "target", data, data, model=DummyClassifier(), output_dir=tmp_path)
