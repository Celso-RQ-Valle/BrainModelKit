"""Pandas-only cross-validation wrappers and all-relevant selection."""

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV, SequentialFeatureSelector

from brainmodelkit.training.pandas import train_model

from ._evaluation import evaluation_frame, finalize_selection
from ._pandas_evaluation import validate_evaluation
from ._pandas_model import estimator, feature_importance_selection, l1_selection
from ._pandas_statistics import supervised
from ._selection import SelectionResult, groups, integer, number, resolve_frame


def rfecv(
    train_df: pd.DataFrame | None = None,
    target_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: pd.DataFrame | None = None,
    oot_df: pd.DataFrame | None = None,
    df_oot: pd.DataFrame | None = None,
    df_scoring: pd.DataFrame | None = None,
    run_name: str = "feature_selection",
    model: Any = "logistic_regression",
    model_params: dict | None = None,
    scoring: str = "roc_auc",
    cv: Any = 5,
    step: int = 1,
    min_features_to_select: int = 1,
    n_jobs: int | None = None,
    random_state: int = 42,
) -> SelectionResult:
    """Choose the feature count by cross-validated recursive elimination.

    See [RFECV](../../docs/feature_selection.md#rfecv) for configuration and
    examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    oot_df = evaluation_frame(oot_df, df_oot, df_scoring)
    validate_evaluation(train_df, oot_df, df_scoring, target_col, feature_cols)
    features = supervised(train_df, target_col, feature_cols)
    integer(step, "step")
    integer(min_features_to_select, "min_features_to_select")
    if min_features_to_select > len(features):
        raise ValueError("min_features_to_select exceeds feature count")
    selector = RFECV(
        estimator(model, model_params, random_state),
        step=step,
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=n_jobs,
    ).fit(train_df[features], train_df[target_col])
    rows = [
        {"feature": f, "ranking": int(rank), "selected": bool(keep)}
        for f, rank, keep in zip(
            features, selector.ranking_, selector.support_, strict=True
        )
    ]
    result = SelectionResult(
        rows,
        "rfecv",
        {"optimal_feature_count": int(selector.n_features_)},
        cv_results=selector.cv_results_,
        model=selector.estimator_,
    )
    return finalize_selection(
        result,
        train_model,
        train_df=train_df,
        oot_df=oot_df,
        target_col=target_col,
        feature_cols=features,
        model=selector.estimator_,
        df_scoring=df_scoring,
        run_name=run_name,
    )


def sequential_selection(
    train_df: pd.DataFrame | None = None,
    target_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: pd.DataFrame | None = None,
    oot_df: pd.DataFrame | None = None,
    df_oot: pd.DataFrame | None = None,
    df_scoring: pd.DataFrame | None = None,
    run_name: str = "feature_selection",
    model: Any = "logistic_regression",
    model_params: dict | None = None,
    direction: str = "forward",
    n_features_to_select: int = 1,
    scoring: str = "roc_auc",
    cv: Any = 5,
    n_jobs: int | None = None,
    random_state: int = 42,
) -> SelectionResult:
    """Perform greedy forward or backward cross-validated subset selection.

    See [Sequential Selection](
    ../../docs/feature_selection.md#sequential-feature-selection) for direction
    options and examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    oot_df = evaluation_frame(oot_df, df_oot, df_scoring)
    validate_evaluation(train_df, oot_df, df_scoring, target_col, feature_cols)
    features = supervised(train_df, target_col, feature_cols)
    selector = SequentialFeatureSelector(
        estimator(model, model_params, random_state),
        direction=direction,
        n_features_to_select=n_features_to_select,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
    ).fit(train_df[features], train_df[target_col])
    result = SelectionResult(
        [
            {"feature": f, "selected": bool(v)}
            for f, v in zip(features, selector.get_support(), strict=True)
        ],
        "sequential_selection",
        {"direction": direction},
    )
    return finalize_selection(
        result,
        train_model,
        train_df=train_df,
        oot_df=oot_df,
        target_col=target_col,
        feature_cols=features,
        model=selector.estimator,
        df_scoring=df_scoring,
        run_name=run_name,
    )


def stability_selection(
    train_df: pd.DataFrame | None = None,
    target_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: pd.DataFrame | None = None,
    oot_df: pd.DataFrame | None = None,
    df_oot: pd.DataFrame | None = None,
    df_scoring: pd.DataFrame | None = None,
    run_name: str = "feature_selection",
    method: str = "feature_importance",
    n_iterations: int = 20,
    sample_fraction: float = 0.8,
    min_frequency: float = 0.8,
    random_state: int = 42,
    group_by: str | Sequence[str] | None = None,
    selector_kwargs: dict | None = None,
) -> SelectionResult:
    """Estimate selection frequencies with stratified bootstrap resampling.

    Each group receives n_iterations fits and equal weight. This is a robustness
    diagnostic, not a formal false-discovery-controlled stability estimator.

    See [Stability Selection](../../docs/feature_selection.md#stability-selection)
    for grouping and examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    oot_df = evaluation_frame(oot_df, df_oot, df_scoring)
    validate_evaluation(train_df, oot_df, df_scoring, target_col, feature_cols)
    features = supervised(train_df, target_col, feature_cols)
    names = groups(train_df, group_by)
    integer(n_iterations, "n_iterations")
    number(sample_fraction, "sample_fraction", maximum=1)
    number(min_frequency, "min_frequency", maximum=1)
    if sample_fraction == 0:
        raise ValueError("sample_fraction must be > 0")
    if method not in ("feature_importance", "l1"):
        raise ValueError("method must be feature_importance or l1")
    options = dict(selector_kwargs or {})
    if "random_state" in options:
        raise ValueError(
            "Pass random_state to stability_selection, not selector_kwargs"
        )
    if method == "feature_importance" and not options:
        options["top_k"] = max(1, len(features) // 2)
    if (
        method == "feature_importance"
        and options.get("threshold") is None
        and options.get("top_k") is None
    ):
        raise ValueError("Specify threshold or top_k in selector_kwargs")
    rng = np.random.RandomState(random_state)
    counts = np.zeros(len(features), dtype=int)
    values, history = [], []
    frames = (
        train_df.groupby(names, dropna=False, observed=True, sort=False)
        if names
        else [(None, train_df)]
    )
    for key, frame in frames:
        supervised(frame, target_col, features)
        for iteration in range(n_iterations):
            seed = int(rng.randint(0, 2**31 - 1))
            parts = [
                part.sample(
                    n=max(1, int(np.ceil(len(part) * sample_fraction))),
                    replace=True,
                    random_state=seed,
                )
                for _, part in frame.groupby(target_col, sort=False)
            ]
            sample = pd.concat(parts, ignore_index=True)
            selector = (
                feature_importance_selection
                if method == "feature_importance"
                else l1_selection
            )
            result = selector(
                sample, target_col, features, random_state=seed, **options
            )
            counts += np.array([r["selected"] for r in result.feature_table], dtype=int)
            values.append([abs(r["importance"]) for r in result.feature_table])
            history.append(
                {
                    "group": key,
                    "iteration": iteration + 1,
                    "selected_features": result.selected_features,
                }
            )
    values = np.asarray(values)
    rows = [
        {
            "feature": f,
            "selection_count": int(counts[i]),
            "selection_frequency": float(counts[i] / len(values)),
            "mean_importance": float(values[:, i].mean()),
            "std_importance": float(values[:, i].std()),
            "selected": bool(counts[i] / len(values) >= min_frequency),
        }
        for i, f in enumerate(features)
    ]
    evaluation_model = result.model
    result = SelectionResult(
        rows,
        "stability_selection",
        {
            "n_fits": len(values),
            "method": method,
            "random_state": random_state,
            "min_frequency": min_frequency,
        },
        history=history,
    )
    return finalize_selection(
        result,
        train_model,
        train_df=train_df,
        oot_df=oot_df,
        target_col=target_col,
        feature_cols=features,
        model=evaluation_model,
        df_scoring=df_scoring,
        run_name=run_name,
    )


def boruta(
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
    n_estimators: int | str = "auto",
    max_iter: int = 100,
    perc: int = 100,
    alpha: float = 0.05,
    random_state: int = 42,
) -> SelectionResult:
    """Run optional Boruta shadow-feature selection.

    See [Boruta](../../docs/feature_selection.md#boruta) for installation,
    statuses, and examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    oot_df = evaluation_frame(oot_df, df_oot, df_scoring)
    validate_evaluation(train_df, oot_df, df_scoring, target_col, feature_cols)
    features = supervised(train_df, target_col, feature_cols)
    try:
        implementation = importlib.import_module("boruta").BorutaPy
    except ImportError as exc:
        raise ImportError(
            "Boruta requires pip install 'BrainModelKit[pandas,boruta]'"
        ) from exc
    fitted = implementation(
        estimator(model, model_params, random_state),
        n_estimators=n_estimators,
        max_iter=max_iter,
        perc=perc,
        alpha=alpha,
        random_state=random_state,
    )
    fitted.fit(train_df[features].to_numpy(), train_df[target_col].to_numpy())
    rows = [
        {
            "feature": f,
            "ranking": int(rank),
            "status": "confirmed"
            if confirmed
            else "tentative"
            if tentative
            else "rejected",
            "selected": bool(confirmed),
        }
        for f, rank, confirmed, tentative in zip(
            features,
            fitted.ranking_,
            fitted.support_,
            fitted.support_weak_,
            strict=True,
        )
    ]
    result = SelectionResult(
        rows, "boruta", {"random_state": random_state}, model=fitted
    )
    return finalize_selection(
        result,
        train_model,
        train_df=train_df,
        oot_df=oot_df,
        target_col=target_col,
        feature_cols=features,
        model=estimator(model, model_params, random_state),
        df_scoring=df_scoring,
        run_name=run_name,
    )
