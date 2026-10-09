"""Feasibility ordering, units, sampling feedback, and selected-trial refitting."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from brainmodelkit.optimization._common import (
    constraint_values,
    make_study,
    metric_values,
    record_trial,
    select_trial,
    validate_options,
)

optuna = pytest.importorskip("optuna")


@pytest.mark.parametrize(
    "objective,direction,target,limit,operator,expected",
    [
        ("oot_ks", "maximize", "train_oot_ks_abs_shift", 0.05, "<=", 2),
        ("train_oot_ks_abs_shift", "minimize", "oot_ks", 0.40, ">=", 1),
    ],
)
def test_both_constraint_examples(
    objective, direction, target, limit, operator, expected
):
    study = make_study(optuna, None, direction, 42, True)
    # Trial 0 has a better objective but is infeasible in both examples.
    pairs = (
        [(0.95, 0.80), (0.44, 0.41), (0.52, 0.48)]
        if direction == "maximize"
        else [(0.39, 0.39), (0.44, 0.41), (0.52, 0.48)]
    )
    for train, oot in pairs:
        trial = study.ask()
        values = metric_values({"ks": train}, {"ks": 0.42}, {"ks": oot})
        record_trial(trial, values, {}, target, limit, operator)
        study.tell(trial, values[objective])
    selected = select_trial(study, direction, True)
    assert selected.number == expected
    assert selected.user_attrs["feasible"]
    assert study.trials[0].system_attrs["constraints"][0] > 0
    assert constraint_values(study.trials[expected])[0] <= 0


def test_no_feasible_trial_prefers_violation_over_objective():
    study = make_study(optuna, None, "maximize", 42, True)
    for train, oot in [(1.0, 0.8), (0.3, 0.2)]:
        trial = study.ask()
        values = metric_values({"ks": train}, {"ks": 0.4}, {"ks": oot})
        record_trial(trial, values, {}, "train_oot_ks_abs_shift", 0.05, "<=")
        study.tell(trial, oot)
    with pytest.warns(RuntimeWarning, match="No feasible trial.*trial 1"):
        selected = select_trial(study, "maximize", True)
    assert selected.number == 1
    assert study.user_attrs["constraint_satisfied"] is False


def test_constraints_affect_sampling_after_startup():
    studies = []
    for constrained in (False, True):
        study = make_study(optuna, None, "minimize", 42, constrained)
        for _ in range(16):
            trial = study.ask()
            score = trial.suggest_float("score", 0.0, 1.0)
            values = metric_values({"ks": score}, {"ks": score}, {"ks": score})
            record_trial(trial, values, {"score": score}, "oot_ks", 0.8, ">=")
            study.tell(trial, score)
        studies.append(study)
    plain, constrained = studies
    assert [t.params for t in plain.trials[:10]] == [
        t.params for t in constrained.trials[:10]
    ]
    assert [t.params for t in plain.trials[10:]] != [
        t.params for t in constrained.trials[10:]
    ]
    assert all("constraints" in t.system_attrs for t in constrained.trials)


@pytest.mark.parametrize("metric", ["auc", "gini", "ks"])
def test_shift_semantics(metric):
    values = metric_values({metric: 0.45}, {metric: 0.35}, {metric: 0.40})
    assert values[metric] == 0.35
    assert values[f"{metric}_shift"] == pytest.approx(0.05)
    assert values[f"train_oot_{metric}_shift"] == pytest.approx(-0.05)
    assert values[f"cv_oot_{metric}_shift"] == pytest.approx(0.05)
    assert values[f"train_oot_{metric}_abs_shift"] == pytest.approx(0.05)
    assert values[f"cv_oot_{metric}_abs_shift"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_metric": "oot_ks"},
        {"target_limit": 0.4},
        {
            "target_metric": "oot_ks",
            "target_limit": float("nan"),
            "target_operator": ">=",
        },
        {"target_metric": "oot_ks", "target_limit": 0.4, "target_operator": "<"},
        {"target_metric": "unknown", "target_limit": 0.4, "target_operator": ">="},
        {"objective_metric": "unknown"},
        {"search_space": {"lgbm_backend": {}}},
    ],
)
def test_invalid_options(kwargs):
    defaults = dict(
        metrics=("ks",),
        objective_metric="ks",
        direction="maximize",
        n_trials=1,
        target_metric=None,
        target_limit=None,
        target_operator=None,
        search_space={"C": {}},
    )
    with pytest.raises(ValueError):
        validate_options(**(defaults | kwargs))


def test_nonfinite_metrics():
    with pytest.raises(ValueError, match=r"non-finite.*train_ks"):
        metric_values({"ks": float("nan")}, {"ks": 0.4}, {"ks": 0.4})


def test_selected_parameters_refit_and_artifact_match(monkeypatch, tmp_path):
    import json

    from sklearn.tree import DecisionTreeClassifier

    import brainmodelkit.optimization.pandas as module

    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame({"x": np.arange(20), "y": [0, 1] * 10})
    monkeypatch.setattr(
        module,
        "cross_validate",
        lambda *a, **k: SimpleNamespace(
            summary=pd.DataFrame([{"metric": "ks", "mean": 0.3}])
        ),
    )
    # Calls alternate OOT and full-training metrics for each candidate.
    scores = iter([0.8, 1.0, 0.4, 0.43, 0.6, 0.64])
    monkeypatch.setattr(module, "_metric_values", lambda *a: {"ks": next(scores)})
    result = module.optimize(
        frame,
        frame,
        "y",
        ["x"],
        DecisionTreeClassifier(random_state=9),
        {"max_depth": lambda t: t.number + 1},
        objective_metric="oot_ks",
        target_metric="train_oot_ks_abs_shift",
        target_limit=0.05,
        target_operator="<=",
        metrics=("ks",),
        n_trials=3,
        save_as="folder",
        save_path=str(tmp_path / "model.pkl"),
    )
    assert result.selected_trial_number == 2
    assert result.best_params == {"max_depth": 3}
    assert result.best_value == 0.6
    assert result.best_model.max_depth == 3
    assert result.constraint_satisfied is True
    assert set(result.trials.status) == {"COMPLETE"}
    assert result.trials.iloc[0].constraint_violation == pytest.approx(0.15)
    from brainmodelkit.persistence import load_model

    loaded = load_model(str(tmp_path / "model.pkl"), "pickle", trusted=True)
    assert loaded.max_depth == 3
    assert (
        json.loads((tmp_path / "model.pkl.metadata.json").read_text())["parameters"][
            "max_depth"
        ]
        == 3
    )
