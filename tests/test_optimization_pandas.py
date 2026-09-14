"""Focused Optuna optimization contracts for the Pandas backend."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from brainmodelkit.optimization.pandas import optimize

optuna = pytest.importorskip("optuna")


@pytest.fixture
def frames():
    x, y = make_classification(
        n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=3
    )
    frame = pd.DataFrame(x, columns=["x1", "x2"])
    frame["target"] = y
    return frame.iloc[:35].copy(), frame.iloc[35:].copy()


def test_optimization_uses_cv_and_returns_diagnostics(frames):
    train, oot = frames
    result = optimize(
        train,
        oot,
        "target",
        ["x1", "x2"],
        "logistic_regression",
        {"C": {"type": "float", "low": 0.1, "high": 1.0}},
        n_trials=2,
        cv=2,
        save_as="none",
        study_name="test-study",
    )
    assert result.best_model is not None
    assert len(result.trials) == 2
    assert {"cv_ks", "oot_ks", "ks_shift", "params"}.issubset(result.trials)
    assert result.best_value == result.study.best_value
    assert all("cv_ks" in trial.user_attrs for trial in result.study.trials)


def test_optimization_rejects_oot_schema(frames):
    train, oot = frames
    with pytest.raises(ValueError, match="oot_df"):
        optimize(
            train,
            oot.drop(columns="target"),
            "target",
            ["x1", "x2"],
            "logistic_regression",
            {"C": {"type": "float", "low": 0.1, "high": 1.0}},
            n_trials=1,
            save_as="none",
        )
