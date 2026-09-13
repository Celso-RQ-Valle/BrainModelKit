"""Estimator-based Pandas selection, without training persistence side effects."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from brainmodelkit.training._common import importance_records
from brainmodelkit.training.pandas import train_model

from ._evaluation import evaluation_frame, finalize_selection
from ._pandas_evaluation import validate_evaluation
from ._pandas_statistics import supervised
from ._selection import SelectionResult, number, ranked, resolve_frame


def estimator(
    model: Any, model_params: dict | None = None, random_state: int = 42
) -> Any:
    if not isinstance(model, str):
        return clone(model).set_params(**(model_params or {}))
    choices = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }
    if model == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("Install BrainModelKit[pandas,lightgbm]") from exc
        choices[model] = LGBMClassifier
    if model not in choices:
        raise ValueError(f"Unknown model: {model}")
    return choices[model](**{"random_state": random_state, **(model_params or {})})


def feature_importance_selection(
    train_df: pd.DataFrame | None = None,
    target_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: pd.DataFrame | None = None,
    oot_df: pd.DataFrame | None = None,
    df_oot: pd.DataFrame | None = None,
    df_scoring: pd.DataFrame | None = None,
    run_name: str = "feature_selection",
    model: Any = "random_forest",
    model_params: dict | None = None,
    threshold: float | None = None,
    top_k: int | None = None,
    random_state: int = 42,
) -> SelectionResult:
    """Fit a classifier; rank magnitude and retain signed coefficients."""
    train_df = resolve_frame(train_df, df, "train_df", "df")
    oot_df = evaluation_frame(oot_df, df_oot, df_scoring)
    validate_evaluation(train_df, oot_df, df_scoring, target_col, feature_cols)
    features = supervised(train_df, target_col, feature_cols)
    fitted = estimator(model, model_params, random_state).fit(
        train_df[features], train_df[target_col]
    )
    values = getattr(fitted, "feature_importances_", None)
    coefficients = getattr(fitted, "coef_", None)
    if values is None and coefficients is not None:
        coefficients = np.asarray(coefficients)
        values = (
            coefficients[0]
            if coefficients.shape[0] == 1
            else np.linalg.norm(coefficients, axis=0)
        )
    if values is None or len(values) != len(features) or not np.isfinite(values).all():
        raise ValueError("model must expose one finite native importance per feature")
    # Reuse training/RFE importance records, then restore input order.
    records = {r["feature"]: r for r in importance_records(features, values)}
    rows = [records[f] for f in features]
    if coefficients is not None:
        for i, row in enumerate(rows):
            row["coefficients"] = coefficients[:, i].tolist()
    result = ranked(
        rows, "importance", "feature_importance", threshold, top_k, absolute=True
    )
    result.model = fitted
    return finalize_selection(
        result,
        train_model,
        train_df=train_df,
        oot_df=oot_df,
        target_col=target_col,
        feature_cols=features,
        model=fitted,
        df_scoring=df_scoring,
        run_name=run_name,
    )


def l1_selection(
    train_df: pd.DataFrame | None = None,
    target_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: pd.DataFrame | None = None,
    oot_df: pd.DataFrame | None = None,
    df_oot: pd.DataFrame | None = None,
    df_scoring: pd.DataFrame | None = None,
    run_name: str = "feature_selection",
    C: float = 1.0,
    coefficient_tolerance: float = 1e-8,
    random_state: int = 42,
    max_iter: int = 1000,
) -> SelectionResult:
    """L1 logistic selection; scale features first. C is inverse regularization."""
    train_df = resolve_frame(train_df, df, "train_df", "df")
    oot_df = evaluation_frame(oot_df, df_oot, df_scoring)
    validate_evaluation(train_df, oot_df, df_scoring, target_col, feature_cols)
    number(C, "C")
    number(coefficient_tolerance, "coefficient_tolerance")
    if C == 0:
        raise ValueError("C must be > 0")
    supervised(train_df, target_col, feature_cols, binary=True)
    result = feature_importance_selection(
        train_df,
        target_col,
        feature_cols,
        model=LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=C,
            max_iter=max_iter,
            random_state=random_state,
        ),
    )
    result.method = "l1"
    result.metadata.update(C=C, coefficient_tolerance=coefficient_tolerance)
    for row in result.feature_table:
        row["selected"] = abs(row["importance"]) > coefficient_tolerance
    return finalize_selection(
        result,
        train_model,
        train_df=train_df,
        oot_df=oot_df,
        target_col=target_col,
        feature_cols=feature_cols,
        model=result.model,
        df_scoring=df_scoring,
        run_name=run_name,
    )


def permutation_importance_selection(
    model: Any,
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    *,
    scoring: str = "roc_auc",
    n_repeats: int = 5,
    random_state: int = 42,
    threshold: float | None = None,
    top_k: int | None = None,
    n_jobs: int | None = None,
) -> SelectionResult:
    """Evaluate a fitted model on held-out data by repeated feature permutation."""
    features = supervised(df, target_col, feature_cols)
    values = permutation_importance(
        model,
        df[features],
        df[target_col],
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    rows = [
        {"feature": f, "importance_mean": float(mean), "importance_std": float(std)}
        for f, mean, std in zip(
            features, values.importances_mean, values.importances_std, strict=True
        )
    ]
    return ranked(rows, "importance_mean", "permutation_importance", threshold, top_k)
