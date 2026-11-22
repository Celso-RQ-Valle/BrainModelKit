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


@pytest.mark.parametrize(
    "space",
    [
        {"C": {"type": "float", "low": 0.1, "high": 1.0}},
        {"C": [0.1, 1.0], "max_iter": [1000]},
    ],
)
def test_optimization_uses_cv_and_returns_diagnostics(frames, space):
    train, oot = frames
    result = optimize(
        train,
        oot,
        "target",
        ["x1", "x2"],
        "logistic_regression",
        space,
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
    if isinstance(space["C"], list):
        assert all(trial.params["C"] in space["C"] for trial in result.study.trials)
        assert result.best_params["max_iter"] == 1000
        assert result.best_params["C"] == result.best_model.C


def test_candidate_lists_preserve_types_and_mix_with_ranges():
    from brainmodelkit.optimization import suggest_parameters

    trial = optuna.trial.FixedTrial({"leaf": 1000, "depth": None, "rate": 0.2})
    params = suggest_parameters(
        trial,
        {
            "leaf": [200, 1000],
            "depth": [None, 3],
            "rate": {"type": "float", "low": 0.1, "high": 0.3},
        },
    )
    assert params == {"leaf": 1000, "depth": None, "rate": 0.2}


@pytest.mark.parametrize("choices", [[], [[200, 1000]], [{"low": 200}]])
def test_invalid_candidate_lists(choices):
    from brainmodelkit.optimization import suggest_parameters

    with pytest.raises(ValueError, match="Candidate list"):
        suggest_parameters(optuna.trial.FixedTrial({}), {"leaf": choices})


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
