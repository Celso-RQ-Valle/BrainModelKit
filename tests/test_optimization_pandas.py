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
    assert result.trials_path is None
    assert result.best_model is not None
    assert len(result.trials) == 2
    assert {"cv_ks", "oot_ks", "ks_shift", "params"}.issubset(result.trials)
    assert result.best_value == result.study.best_value
    assert all("cv_ks" in trial.user_attrs for trial in result.study.trials)
    if isinstance(space["C"], list):
        assert all(0.1 <= trial.params["C"] <= 1.0 for trial in result.study.trials)
        assert any(0.1 < trial.params["C"] < 1.0 for trial in result.study.trials)
        assert result.best_params["max_iter"] == 1000
        assert result.best_model.max_iter == 1000
        assert result.best_params["C"] == result.best_model.C


def test_intervals_sample_interior_values_and_preserve_types():
    from brainmodelkit.optimization import suggest_parameters

    trial = optuna.trial.FixedTrial(
        {"leaf": 537, "depth": None, "rate": 0.2, "fixed": 100, "mixed": 1.5}
    )
    params = suggest_parameters(
        trial,
        {
            "leaf": [200, 1000],
            "depth": {"type": "categorical", "choices": [None, 3]},
            "fixed": [100],
            "mixed": [1, 2.0],
            "rate": [0.1, 0.3],
        },
    )
    assert params == {
        "leaf": 537,
        "depth": None,
        "rate": 0.2,
        "fixed": 100,
        "mixed": 1.5,
    }
    assert isinstance(trial.distributions["leaf"], optuna.distributions.IntDistribution)
    assert isinstance(
        trial.distributions["rate"], optuna.distributions.FloatDistribution
    )
    assert isinstance(
        trial.distributions["mixed"], optuna.distributions.FloatDistribution
    )


@pytest.mark.parametrize(
    "choices",
    [
        [],
        [float("inf")],
        [float("nan")],
        [15, 31, 63],
        [1000, 200],
        [True, 2],
        [False, 1],
        [0, True],
        [False, True, False],
        [None, 3],
        ["a", "b"],
        [0, float("inf")],
        [float("nan"), 1],
        [[200, 1000]],
        [{"low": 200}],
    ],
)
def test_invalid_intervals(choices):
    from brainmodelkit.optimization import suggest_parameters

    with pytest.raises(ValueError, match="Interval"):
        suggest_parameters(optuna.trial.FixedTrial({}), {"leaf": choices})


@pytest.mark.parametrize("value", [100, 0.1, "gini", True, False, None])
def test_fixed_singletons_are_recorded_and_preserve_types(value):
    from brainmodelkit.optimization import suggest_parameters

    study = optuna.create_study()
    trial = study.ask()
    params = suggest_parameters(trial, {"setting": [value]})
    study.tell(trial, 1.0)
    assert params == {"setting": value}
    assert type(params["setting"]) is type(value)
    assert study.trials[0].params == params


def test_equal_bounds_remain_backward_compatible():
    from brainmodelkit.optimization import suggest_parameters

    assert suggest_parameters(
        optuna.trial.FixedTrial({"size": 100}), {"size": [100, 100]}
    ) == {"size": 100}


@pytest.mark.parametrize("choices", [[False, True], [True, False]])
@pytest.mark.parametrize("value", [False, True])
def test_boolean_choices_preserve_type_and_distribution(choices, value):
    from brainmodelkit.optimization import suggest_parameters

    trial = optuna.trial.FixedTrial({"fit_intercept": value})
    params = suggest_parameters(trial, {"fit_intercept": choices})
    assert params["fit_intercept"] is value
    distribution = trial.distributions["fit_intercept"]
    assert isinstance(distribution, optuna.distributions.CategoricalDistribution)
    assert distribution.choices == tuple(choices)


def test_boolean_choices_reach_fitted_model(frames):
    train, validation = frames
    result = optimize(
        train, validation, "target", ["x1", "x2"], "logistic_regression",
        {"fit_intercept": [False, True]}, n_trials=2, cv=2, save_as="none",
    )
    assert type(result.best_params["fit_intercept"]) is bool
    assert result.best_model.fit_intercept is result.best_params["fit_intercept"]
    assert all(type(t.params["fit_intercept"]) is bool for t in result.study.trials)


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


@pytest.mark.parametrize("constrained", [False, True])
def test_save_trials_csv_with_parameters_and_metrics(frames, tmp_path, constrained):
    import json

    train, validation = frames
    path = tmp_path / "reports" / "trials.csv"
    options = (
        {
            "target_metric": "train_oot_ks_abs_shift",
            "target_limit": 1.0,
            "target_operator": "<=",
        }
        if constrained
        else {}
    )
    result = optimize(
        train,
        validation,
        "target",
        ["x1", "x2"],
        "logistic_regression",
        {"C": [0.1, 1.0], "max_iter": [1000]},
        n_trials=3,
        cv=2,
        save_as="none",
        save_trials_path=path,
        **options,
    )
    saved = pd.read_csv(path)
    assert result.trials_path == str(path)
    assert len(saved) == 3
    assert saved.selected.sum() == 1
    chosen = saved.loc[saved.selected].iloc[0]
    assert chosen.trial == result.selected_trial_number
    assert json.loads(chosen.params) == result.best_params
    assert set(saved.status) == {"COMPLETE"}
    assert saved.param_C.between(0.1, 1.0).all()
    assert saved.param_max_iter.eq(1000).all()
    for metric in [
        "train_auc",
        "cv_auc",
        "oot_auc",
        "train_oot_ks_abs_shift",
        "constraint_violation",
        "value",
    ]:
        assert saved[metric].tolist() == pytest.approx(result.trials[metric].tolist())
    assert saved.feasible.all()


def test_invalid_export_path_fails_before_trials(frames, monkeypatch):
    import brainmodelkit.optimization.pandas as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Should validate the destination before fitting")

    monkeypatch.setattr(module, "cross_validate", forbidden)
    train, validation = frames
    with pytest.raises(ValueError, match=r"\.csv"):
        optimize(
            train,
            validation,
            "target",
            ["x1", "x2"],
            "logistic_regression",
            {"C": [0.1, 1.0]},
            save_trials_path="report.txt",
        )
