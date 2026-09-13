"""Pandas/scikit-learn classification statistics; no Spark dependencies."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.utils.multiclass import check_classification_targets

from ._pandas_quality import nonempty, numeric
from ._selection import (
    SelectionResult,
    columns,
    correlation_result,
    integer,
    number,
    ranked,
    resolve_frame,
)


def supervised(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    *,
    binary: bool = False,
    numeric_features: bool = True,
) -> list[str]:
    features = columns(df, feature_cols, target_col)
    nonempty(df)
    y = df[target_col]
    if y.isna().any() or y.nunique() < 2:
        raise ValueError("Target must be non-null and contain at least two classes")
    if binary and not y.isin([0, 1]).all():
        raise ValueError("Target must contain binary 0/1 values")
    check_classification_targets(y)
    if numeric_features:
        numeric(df, features)
    return features


def correlation_filter(
    train_df: pd.DataFrame | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: pd.DataFrame | None = None,
    threshold: float = 0.9,
    method: str = "pearson",
) -> SelectionResult:
    """Greedy absolute-correlation filter; earlier retained columns win ties.

    See [Correlation](../../docs/feature_selection.md#correlation) for method
    semantics and examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = columns(train_df, feature_cols)
    number(threshold, "threshold", maximum=1)
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be pearson or spearman")
    numeric(train_df, features)
    if len(train_df) < 2:
        raise ValueError("Correlation requires at least two rows")
    return correlation_result(
        features, train_df[features].corr(method=method).to_numpy(), threshold, method
    )


def mutual_information(
    train_df: pd.DataFrame | None = None,
    target_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: pd.DataFrame | None = None,
    threshold: float | None = None,
    top_k: int | None = None,
    random_state: int = 42,
    discrete_features: bool | Sequence[bool] = False,
    n_neighbors: int = 3,
) -> SelectionResult:
    """Estimate classification mutual information in nats.

    See [Mutual Information](../../docs/feature_selection.md#mutual-information)
    for discrete-input handling and examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = supervised(train_df, target_col, feature_cols)
    integer(n_neighbors, "n_neighbors")
    values = mutual_info_classif(
        train_df[features],
        train_df[target_col],
        discrete_features=discrete_features,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    rows = [
        {"feature": f, "mutual_information": float(v)}
        for f, v in zip(features, values, strict=True)
    ]
    return ranked(rows, "mutual_information", "mutual_information", threshold, top_k)


def chi_square(
    train_df: pd.DataFrame | None = None,
    target_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: pd.DataFrame | None = None,
    max_p_value: float | None = 0.05,
    top_k: int | None = None,
) -> SelectionResult:
    """Run chi-square selection for nonnegative integer count features.

    See [Chi-Square](../../docs/feature_selection.md#chi-square) for Pandas
    semantics, p-value filtering, and examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = supervised(train_df, target_col, feature_cols)
    x = train_df[features].to_numpy(dtype=float)
    if (x < 0).any() or (x != np.floor(x)).any():
        raise ValueError(
            "Chi-Square requires nonnegative integer counts or indicators; "
            "bin/encode continuous inputs"
        )
    statistic, p_values = chi2(x, train_df[target_col])
    result = test_result(
        features, statistic, p_values, "statistic", "chi_square", max_p_value, top_k
    )
    for row in result.feature_table:
        row["selected"] = row["selected"] and train_df[row["feature"]].nunique() > 1
    return result


def test_result(
    features: list[str],
    values: Sequence[float],
    p_values: Sequence[float],
    score: str,
    method: str,
    max_p_value: float | None,
    top_k: int | None,
) -> SelectionResult:
    if max_p_value is not None:
        number(max_p_value, "max_p_value", maximum=1)
    rows = [
        {"feature": f, score: float(v), "p_value": float(p)}
        for f, v, p in zip(features, values, p_values, strict=True)
    ]
    result = ranked(rows, score, method, top_k=top_k)
    for row in rows:
        row["selected"] = bool(
            row["selected"]
            and np.isfinite(row["p_value"])
            and (max_p_value is None or row["p_value"] <= max_p_value)
        )
    result.metadata["max_p_value"] = max_p_value
    return result


def anova(
    train_df: pd.DataFrame | None = None,
    target_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: pd.DataFrame | None = None,
    max_p_value: float | None = 0.05,
    top_k: int | None = None,
) -> SelectionResult:
    """Run a one-way classification ANOVA F-test.

    See [ANOVA](../../docs/feature_selection.md#anova--f-test) for assumptions
    and examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = supervised(train_df, target_col, feature_cols)
    if len(train_df) <= train_df[target_col].nunique():
        raise ValueError("ANOVA requires residual degrees of freedom")
    values, p_values = f_classif(train_df[features], train_df[target_col])
    return test_result(
        features, values, p_values, "f_statistic", "anova", max_p_value, top_k
    )


def information_value(
    train_df: pd.DataFrame | None = None,
    target_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: pd.DataFrame | None = None,
    n_bins: int = 10,
    min_iv: float | None = None,
    smoothing: float = 0.5,
    binning: str = "quantile",
) -> SelectionResult:
    """Compute smoothed binary Information Value with a missing bin.

    See [Information Value](../../docs/feature_selection.md#information-value)
    for binning, WoE diagnostics, and examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = supervised(
        train_df, target_col, feature_cols, binary=True, numeric_features=False
    )
    integer(n_bins, "n_bins", 2)
    number(smoothing, "smoothing")
    if smoothing == 0:
        raise ValueError("smoothing must be > 0")
    if binning not in ("quantile", "uniform"):
        raise ValueError("binning must be quantile or uniform")
    rows, details, edges_by_feature = [], [], {}
    for f in features:
        series = train_df[f]
        edges = []
        if pd.api.types.is_numeric_dtype(series):
            numeric(train_df, [f], allow_missing=True)
            valid = series.dropna().astype(float)
            if len(valid) and valid.nunique() > 1:
                edges = (
                    np.unique(valid.quantile(np.linspace(0, 1, n_bins + 1)).to_numpy())[
                        1:-1
                    ]
                    if binning == "quantile"
                    else np.linspace(valid.min(), valid.max(), n_bins + 1)[1:-1]
                ).tolist()
            codes = pd.Series(
                np.searchsorted(
                    edges, series.fillna(0).to_numpy(dtype=float), side="left"
                ),
                index=train_df.index,
            )
            codes = codes.where(series.notna(), -1)
        else:
            codes = pd.Series(pd.factorize(series, sort=False)[0], index=train_df.index)
        edges_by_feature[f] = edges
        frame = pd.DataFrame(
            {"bin": codes.to_numpy(), "event": train_df[target_col].to_numpy()}
        )
        counts = frame.groupby("bin", sort=True)["event"].agg(["sum", "count"])
        k = len(counts)
        total_event = float(train_df[target_col].sum())
        iv = 0.0
        for code, row in counts.iterrows():
            event, non_event = float(row["sum"]), float(row["count"] - row["sum"])
            de = (event + smoothing) / (total_event + smoothing * k)
            dn = (non_event + smoothing) / (len(train_df) - total_event + smoothing * k)
            woe = float(np.log(de / dn))
            contribution = (de - dn) * woe
            label = (
                None
                if code == -1
                else (
                    str(series[codes == code].iloc[0])
                    if not pd.api.types.is_numeric_dtype(series)
                    else str(int(code))
                )
            )
            details.append(
                {
                    "feature": f,
                    "bin": int(code),
                    "label": label,
                    "is_missing": code == -1,
                    "event_count": event,
                    "non_event_count": non_event,
                    "distribution_event": de,
                    "distribution_non_event": dn,
                    "woe": woe,
                    "iv_component": contribution,
                }
            )
            iv += contribution
        rows.append({"feature": f, "iv": iv})
    result = ranked(rows, "iv", "information_value", min_iv)
    result.bin_details = pd.DataFrame(details)
    result.metadata.update(
        n_bins=n_bins, binning=binning, smoothing=smoothing, bin_edges=edges_by_feature
    )
    return result
