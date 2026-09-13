"""Pandas/scikit-learn cross-validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from brainmodelkit.metrics.pandas import _calculate_auc_gini, calculate_ks

from . import CrossValidationResult


def _estimator(model: Any, params: dict | None, random_state: int | None):
    if isinstance(model, str):
        from brainmodelkit.feature_selection._pandas_model import estimator

        return estimator(model, params, 42 if random_state is None else random_state)
    return clone(model).set_params(**(params or {}))


def _validate(df, target_col, feature_cols):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    features = list(feature_cols)
    if not features or len(set(features)) != len(features) or target_col in features:
        raise ValueError(
            "feature_cols must be unique, non-empty, and exclude target_col"
        )
    missing = sorted(set([*features, target_col]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    target = df[target_col]
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError(f"`{target_col}` must contain only binary values 0 and 1.")
    if target.nunique() != 2:
        raise ValueError("Cross-validation requires both target classes")
    return features


def cross_validate(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    model: Any,
    model_params: dict | None = None,
    cv: int = 5,
    strategy: str = "stratified",
    group_col: str | None = None,
    date_col: str | None = None,
    metrics: Sequence[str] = ("ks", "auc", "gini"),
    return_train_metrics: bool = False,
    random_state: int | None = 42,
    return_models: bool = False,
) -> CrossValidationResult:
    """Fit and evaluate one clone of ``model`` per development-data fold."""
    features = _validate(df, target_col, feature_cols)
    if not isinstance(cv, int) or isinstance(cv, bool) or cv < 2:
        raise ValueError("cv must be an integer >= 2")
    requested = tuple(str(m).lower() for m in metrics)
    if not requested or any(m not in {"ks", "auc", "gini"} for m in requested):
        raise ValueError("metrics must contain only ks, auc, and gini")
    if group_col and group_col not in df.columns:
        raise ValueError(f"Missing columns: ['{group_col}']")
    if date_col and date_col not in df.columns:
        raise ValueError(f"Missing columns: ['{date_col}']")
    if group_col and date_col:
        raise ValueError("group_col and date_col cannot be combined")
    if strategy == "group" or group_col:
        if not group_col:
            raise ValueError("group_col is required for group strategy")
        splitter = GroupKFold(n_splits=cv)
        split_args = (df[features], df[target_col], df[group_col])
    elif strategy in {"time", "time_series", "ordered"} or date_col:
        if not date_col:
            raise ValueError("date_col is required for time strategy")
        order = df[date_col].sort_values(kind="stable").index
        ordered = df.loc[order]
        splitter = TimeSeriesSplit(n_splits=cv)
        split_args = (ordered[features], ordered[target_col])
    elif strategy == "kfold":
        splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        split_args = (df[features], df[target_col])
    elif strategy == "stratified":
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        split_args = (df[features], df[target_col])
    else:
        raise ValueError("strategy must be stratified, kfold, group, or time_series")

    records, models = [], [] if return_models else None
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(*split_args), 1):
        frame = df.loc[split_args[0].index]
        train = frame.iloc[train_idx]
        valid = frame.iloc[valid_idx]
        if train[target_col].nunique() < 2:
            values = {"fold": fold, **{m.upper(): float("nan") for m in requested}}
            if return_train_metrics:
                values.update({f"train_{m.upper()}": float("nan") for m in requested})
            records.append(values)
            if return_models:
                models.append(None)
            continue
        fitted = _estimator(model, model_params, random_state).fit(
            train[features], train[target_col]
        )
        if not hasattr(fitted, "predict_proba"):
            raise TypeError("model must implement predict_proba")
        classes = list(fitted.classes_)
        if 1 not in classes:
            score = np.full(len(valid), np.nan)
            train_score = np.full(len(train), np.nan)
        else:
            pos = classes.index(1)
            score = fitted.predict_proba(valid[features])[:, pos]
            train_score = fitted.predict_proba(train[features])[:, pos]
        values = {"fold": fold}
        for metric in requested:
            values[metric.upper()] = (
                {
                    "ks": calculate_ks,
                    "auc": lambda y, s: _calculate_auc_gini(y, s)[0],
                    "gini": lambda y, s: _calculate_auc_gini(y, s)[1],
                }[metric]
            )(valid[target_col], pd.Series(score, index=valid.index))
        if return_train_metrics:
            for metric in requested:
                values[f"train_{metric.upper()}"] = (
                    {
                        "ks": calculate_ks,
                        "auc": lambda y, s: _calculate_auc_gini(y, s)[0],
                        "gini": lambda y, s: _calculate_auc_gini(y, s)[1],
                    }[metric]
                )(train[target_col], pd.Series(train_score, index=train.index))
        records.append(values)
        if return_models:
            models.append(fitted)
    fold_metrics = pd.DataFrame(records)
    summary_records = []
    for metric in [m.upper() for m in requested] + (
        [f"train_{m.upper()}" for m in requested] if return_train_metrics else []
    ):
        series = fold_metrics[metric].astype(float)
        summary_records.append(
            {
                "metric": metric,
                "mean": series.mean(),
                "std": series.std(ddof=0),
                "min": series.min(),
                "max": series.max(),
            }
        )
    return CrossValidationResult(fold_metrics, pd.DataFrame(summary_records), models)


__all__ = ["cross_validate"]
