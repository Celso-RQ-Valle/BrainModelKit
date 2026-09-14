"""Optuna optimization using Pandas and scikit-learn only."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from brainmodelkit.metrics.pandas import _calculate_auc_gini, calculate_ks
from brainmodelkit.model_selection.pandas import _estimator, cross_validate
from brainmodelkit.training.pandas import train_model

from . import OptimizationResult, destination, require_optuna, suggest_parameters


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
) -> OptimizationResult:
    """Optimize parameters on CV and report, but never optimize on, OOT."""
    optuna = require_optuna()
    features = list(feature_cols)
    if not features or len(set(features)) != len(features) or target_col in features:
        raise ValueError(
            "feature_cols must be unique, non-empty, and exclude target_col"
        )
    _validate(train_df, target_col, features, "train_df")
    _validate(oot_df, target_col, features, "oot_df")
    requested = tuple(str(metric).lower() for metric in metrics)
    if any(metric not in {"ks", "auc", "gini"} for metric in requested):
        raise ValueError("metrics must contain only ks, auc, and gini")
    objective_metric = objective_metric.lower()
    if objective_metric not in requested:
        raise ValueError("objective_metric must be included in metrics")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be maximize or minimize")
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise ValueError("n_trials must be an integer >= 1")

    def objective(trial):
        params = suggest_parameters(trial, search_space)
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
        for metric in requested:
            trial.set_user_attr(f"cv_{metric}", cv_values[metric])
            trial.set_user_attr(f"oot_{metric}", oot_values[metric])
            trial.set_user_attr(
                f"{metric}_shift", oot_values[metric] - cv_values[metric]
            )
        trial.set_user_attr("parameters", params)
        return cv_values[objective_metric]

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials)
    rows = []
    for trial in study.trials:
        row = {"trial": trial.number, "params": trial.params, "value": trial.value}
        for metric in requested:
            row.update(
                {
                    f"cv_{metric}": trial.user_attrs.get(f"cv_{metric}"),
                    f"oot_{metric}": trial.user_attrs.get(f"oot_{metric}"),
                    f"{metric}_shift": trial.user_attrs.get(f"{metric}_shift"),
                }
            )
        rows.append(row)
    final = train_model(
        study_name or "optimization",
        target_col,
        features,
        train_df,
        oot_df,
        model=model,
        model_params=study.best_params,
        save_model_to=destination(save_as, save_path),
        save_path=save_path,
    )
    return OptimizationResult(
        study,
        dict(study.best_params),
        float(study.best_value),
        pd.DataFrame(rows),
        final.model,
    )


__all__ = ["optimize"]
