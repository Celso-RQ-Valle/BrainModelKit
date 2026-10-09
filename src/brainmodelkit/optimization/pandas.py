"""Optuna optimization using Pandas and scikit-learn only."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from brainmodelkit.metrics.pandas import _calculate_auc_gini, calculate_ks
from brainmodelkit.model_selection.pandas import _estimator, cross_validate
from brainmodelkit.training.pandas import train_model

from . import OptimizationResult, destination, require_optuna, suggest_parameters
from ._common import (
    make_study,
    metric_values,
    record_trial,
    select_trial,
    trial_records,
    validate_options,
)


def _metric_values(target, score, names):
    auc, gini = _calculate_auc_gini(target, score)
    values = {"ks": calculate_ks(target, score), "auc": auc, "gini": gini}
    return {name: float(values[name]) for name in names}


def _validate(df, target, features, label):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    missing = sorted(set([*features, target]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {label}: {missing}")
    if df[target].isna().any() or not df[target].isin([0, 1]).all():
        raise ValueError(f"`{target}` must contain only binary values 0 and 1.")


def optimize(
    train_df: pd.DataFrame,
    oot_df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    model: Any,
    search_space: dict[str, Any],
    objective_metric: str = "ks",
    direction: str = "maximize",
    n_trials: int = 50,
    cv: int = 5,
    metrics: Sequence[str] = ("ks", "auc", "gini"),
    save_as: str | None = None,
    save_path: str | None = None,
    study_name: str | None = None,
    random_state: int = 42,
    target_metric: str | None = None,
    target_limit: float | None = None,
    target_operator: str | None = None,
) -> OptimizationResult:
    """Tune with CV/train/OOT metrics; OOT used for selection is validation data."""
    optuna = require_optuna()
    save_model_to = destination(save_as, save_path)
    features = list(feature_cols)
    if not features or len(set(features)) != len(features) or target_col in features:
        raise ValueError(
            "feature_cols must be unique, non-empty, and exclude target_col"
        )
    _validate(train_df, target_col, features, "train_df")
    _validate(oot_df, target_col, features, "oot_df")
    requested, objective_metric, target_metric = validate_options(
        metrics,
        objective_metric,
        direction,
        n_trials,
        target_metric,
        target_limit,
        target_operator,
        search_space,
    )

    def objective(trial):
        params = suggest_parameters(trial, search_space)
        trial.set_user_attr("parameters", params)
        cv_result = cross_validate(
            train_df,
            target_col,
            features,
            model,
            model_params=params,
            cv=cv,
            metrics=requested,
            random_state=random_state,
        )
        cv_values = {
            row["metric"].lower(): float(row["mean"])
            for _, row in cv_result.summary.iterrows()
        }
        fitted = _estimator(model, params, random_state).fit(
            train_df[features], train_df[target_col]
        )
        positive = list(fitted.classes_).index(1)
        score = fitted.predict_proba(oot_df[features])[:, positive]
        oot_values = _metric_values(oot_df[target_col], score, requested)
        train_score = fitted.predict_proba(train_df[features])[:, positive]
        train_values = _metric_values(train_df[target_col], train_score, requested)
        values = metric_values(train_values, cv_values, oot_values)
        record_trial(
            trial, values, params, target_metric, target_limit, target_operator
        )
        return values[objective_metric]

    study = make_study(
        optuna, study_name, direction, random_state, target_metric is not None
    )
    study.set_user_attr("objective_metric", objective_metric)
    if target_metric is not None:
        study.set_user_attr(
            "constraint",
            {
                "metric": target_metric,
                "limit": target_limit,
                "operator": target_operator,
            },
        )
    study.optimize(objective, n_trials=n_trials)
    selected = select_trial(study, direction, target_metric is not None)
    selected_params = dict(selected.user_attrs["parameters"])
    rows = trial_records(study)
    final = train_model(
        study_name or "optimization",
        target_col,
        features,
        train_df,
        oot_df,
        model=_estimator(model, selected_params, random_state),
        model_params=None,
        save_model_to=save_model_to,
        save_path=save_path,
    )
    return OptimizationResult(
        study,
        selected_params,
        float(selected.value),
        pd.DataFrame(rows),
        final.model,
        selected_trial_number=selected.number,
        constraint_satisfied=selected.user_attrs["feasible"] if target_metric else None,
    )


__all__ = ["optimize"]
