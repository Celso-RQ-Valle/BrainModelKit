"""Optuna optimization using native Spark ML and distributed data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from brainmodelkit.metrics.pyspark import auc_gini, ks_ntile
from brainmodelkit.model_selection.pyspark import _pipeline, cross_validate
from brainmodelkit.training._spark_estimators import validate_backend
from brainmodelkit.training.pyspark import train_model

from . import OptimizationResult, destination, require_optuna, suggest_parameters
from ._common import (
    make_study,
    metric_values,
    record_trial,
    select_trial,
    trial_records,
    validate_options,
)


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
    target_metric: str | None = None,
    target_limit: float | None = None,
    target_operator: str | None = None,
    lgbm_backend: str = "default",
) -> OptimizationResult:
    """Tune with CV/train/OOT metrics; OOT used for selection is validation data."""
    validate_backend(lgbm_backend)
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

    from pyspark.ml.functions import vector_to_array

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
            n_tiles=n_tiles,
            lgbm_backend=lgbm_backend,
        )
        cv_values = {
            row["metric"].lower(): float(row["mean"]) for row in cv_result.summary
        }
        fitted = _pipeline(
            features, target_col, model, params, random_state, lgbm_backend
        ).fit(train_df)
        scored = fitted.transform(oot_df).withColumn(
            "score", vector_to_array("probability")[1]
        )
        oot_values = _metrics(scored, target_col, requested, n_tiles)
        train_scored = fitted.transform(train_df).withColumn(
            "score", vector_to_array("probability")[1]
        )
        train_values = _metrics(train_scored, target_col, requested, n_tiles)
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
        model=model,
        model_params=selected_params,
        save_model_to=save_model_to,
        save_path=save_path,
        n_tiles=n_tiles,
        lgbm_backend=lgbm_backend,
        random_state=random_state,
    )
    return OptimizationResult(
        study,
        selected_params,
        float(selected.value),
        rows,
        final.model,
        selected_trial_number=selected.number,
        constraint_satisfied=selected.user_attrs["feasible"] if target_metric else None,
    )


__all__ = ["optimize"]
