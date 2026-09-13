"""Pandas-only cross-validation wrappers and all-relevant selection.

Usage, method algorithms, parameters, and examples:
https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection.md
"""

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from brainmodelkit.model_selection.pandas import cross_validate
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

    Usage
    -----
    Import `brainmodelkit.feature_selection.pandas` as `fs`, then call:

        result = fs.rfecv(
            train_df, "target", ["income", "age"], cv=3
        )
        print(result.selected_features)

    Parameters, return fields, algorithm, assumptions, and complete examples:
    https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/rfecv.md
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    oot_df = evaluation_frame(oot_df, df_oot, df_scoring)
    validate_evaluation(train_df, oot_df, df_scoring, target_col, feature_cols)
    features = supervised(train_df, target_col, feature_cols)
    integer(step, "step")
    integer(min_features_to_select, "min_features_to_select")
    if min_features_to_select > len(features):
        raise ValueError("min_features_to_select exceeds feature count")
    current = features.copy()
    paths = []
    while True:
        fitted = estimator(model, model_params, random_state).fit(
            train_df[current], train_df[target_col]
        )
        cv_result = cross_validate(
            train_df,
            target_col,
            current,
            model,
            model_params=model_params,
            cv=cv,
            metrics=("auc",),
            random_state=random_state,
        )
        score_values = cv_result.fold_metrics["AUC"].to_numpy()
        paths.append(
            (
                current.copy(),
                float(np.nanmean(score_values)),
                float(np.nanstd(score_values)),
            )
        )
        if len(current) == min_features_to_select:
            break
        values = getattr(fitted, "feature_importances_", None)
        if values is None and hasattr(fitted, "coef_"):
            values = np.abs(np.asarray(fitted.coef_)).ravel()
        if values is None:
            raise ValueError("model must expose native importances for rfecv")
        remove = sorted(zip(current, np.abs(values), strict=True), key=lambda x: x[1])[
            : min(step, len(current) - min_features_to_select)
        ]
        current = [f for f in current if f not in {name for name, _ in remove}]
    best = max(paths, key=lambda x: (x[1], -len(x[0])))
    selected = set(best[0])
    ranks = {f: 1 if f in selected else 2 for f in features}
    rows = [
        {"feature": f, "ranking": ranks[f], "selected": f in selected} for f in features
    ]
    selector_model = estimator(model, model_params, random_state).fit(
        train_df[best[0]], train_df[target_col]
    )
    result = SelectionResult(
        rows,
        "rfecv",
        {"optimal_feature_count": len(best[0])},
        cv_results={
            "n_features": [len(p[0]) for p in paths],
            "mean_test_score": [p[1] for p in paths],
            "std_test_score": [p[2] for p in paths],
        },
        model=selector_model,
    )
    return finalize_selection(
        result,
        train_model,
        train_df=train_df,
        oot_df=oot_df,
        target_col=target_col,
        feature_cols=features,
        model=selector_model,
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

    Usage
    -----
    Import `brainmodelkit.feature_selection.pandas` as `fs`, then call:

        result = fs.sequential_selection(
            train_df, "target", ["income", "age"], cv=3
        )
        print(result.selected_features)

    Parameters, return fields, algorithm, assumptions, and complete examples:
    https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/sequential_selection.md
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    oot_df = evaluation_frame(oot_df, df_oot, df_scoring)
    validate_evaluation(train_df, oot_df, df_scoring, target_col, feature_cols)
    features = supervised(train_df, target_col, feature_cols)
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be forward or backward")
    if (
        not isinstance(n_features_to_select, int)
        or n_features_to_select < 1
        or n_features_to_select > len(features)
    ):
        raise ValueError("n_features_to_select must be between 1 and feature count")
    current = [] if direction == "forward" else features.copy()
    while len(current) != n_features_to_select:
        candidates = []
        for feature in features:
            if (direction == "forward") == (feature in current):
                continue
            subset = (
                [*current, feature]
                if direction == "forward"
                else [f for f in current if f != feature]
            )
            cv_result = cross_validate(
                train_df,
                target_col,
                subset,
                model,
                model_params=model_params,
                cv=cv,
                metrics=("auc",),
                random_state=random_state,
            )
            candidates.append(
                (float(cv_result.fold_metrics["AUC"].mean()), feature, subset)
            )
        _, _, current = max(candidates, key=lambda x: x[0])
    selector_model = estimator(model, model_params, random_state).fit(
        train_df[current], train_df[target_col]
    )
    result = SelectionResult(
        [{"feature": f, "selected": f in current} for f in features],
        "sequential_selection",
        {"direction": direction},
        model=selector_model,
    )
    return finalize_selection(
        result,
        train_model,
        train_df=train_df,
        oot_df=oot_df,
        target_col=target_col,
        feature_cols=features,
        model=selector_model,
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

    Usage
    -----
    Import `brainmodelkit.feature_selection.pandas` as `fs`, then call:

        result = fs.stability_selection(
            train_df, "target", ["income", "age"], n_iterations=5
        )
        print(result.selected_features)

    Parameters, return fields, algorithm, assumptions, and complete examples:
    https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/stability_selection.md
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

    Usage
    -----
    Import `brainmodelkit.feature_selection.pandas` as `fs`, then call:

        result = fs.boruta(
            train_df, "target", ["income", "age"], n_estimators=30, max_iter=30
        )
        print(result.selected_features)

    Parameters, return fields, algorithm, assumptions, and complete examples:
    https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/boruta.md
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
