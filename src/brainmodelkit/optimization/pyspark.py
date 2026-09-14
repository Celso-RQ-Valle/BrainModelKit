"""Optuna optimization using native Spark ML and distributed data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from brainmodelkit.metrics.pyspark import auc_gini, ks_ntile
from brainmodelkit.model_selection.pyspark import _pipeline, cross_validate
from brainmodelkit.training.pyspark import train_model

from . import OptimizationResult, destination, require_optuna, suggest_parameters


def _validate(df, target, features, label):
    missing = sorted(set([*features, target]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {label}: {missing}")
    from pyspark.sql import functions as F

    if df.filter(F.col(target).isNull() | ~F.col(target).isin(0, 1)).limit(1).count():
        raise ValueError(f"`{target}` must contain only binary values 0 and 1.")


def _metrics(scored, target, names, n_tiles):
    values = {}
    if "ks" in names:
        values["ks"] = float(
            ks_ntile(dataframe=scored, target=target, n_tiles=n_tiles).first()["KS"]
        )
    if "auc" in names or "gini" in names:
        row = auc_gini(df=scored, target_column=target, n_tiles=n_tiles).first()
        if "auc" in names:
            values["auc"] = float(row["auc"])
        if "gini" in names:
            values["gini"] = float(row["gini"])
    return values


def optimize(
    train_df,
    oot_df,
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
    n_tiles: int = 10,
) -> OptimizationResult:
    """Optimize on distributed CV; OOT is evaluated only for diagnostics."""
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

    from pyspark.ml.functions import vector_to_array

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
            n_tiles=n_tiles,
        )
        cv_values = {
            row["metric"].lower(): float(row["mean"]) for row in cv_result.summary
        }
        fitted = _pipeline(features, target_col, model, params, random_state).fit(
            train_df
        )
        scored = fitted.transform(oot_df).withColumn(
            "score", vector_to_array("probability")[1]
        )
        oot_values = _metrics(scored, target_col, requested, n_tiles)
        for metric in requested:
            trial.set_user_attr(f"cv_{metric}", cv_values[metric])
            trial.set_user_attr(f"oot_{metric}", oot_values[metric])
            trial.set_user_attr(
                f"{metric}_shift", oot_values[metric] - cv_values[metric]
            )
        trial.set_user_attr("parameters", params)
        return cv_values[objective_metric]

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=random_state),
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
        n_tiles=n_tiles,
    )
    return OptimizationResult(
        study, dict(study.best_params), float(study.best_value), rows, final.model
    )


__all__ = ["optimize"]
