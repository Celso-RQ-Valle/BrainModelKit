"""Pandas model metrics."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def calculate_ks(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculate the exact point-by-point Kolmogorov-Smirnov statistic."""
    paired = pd.DataFrame({"target": y_true, "score": y_pred}).dropna()
    if paired.empty:
        return float("nan")

    event_count = int((paired["target"] == 1).sum())
    non_event_count = int((paired["target"] == 0).sum())
    if event_count == 0 or non_event_count == 0:
        return float("nan")

    distribution = (
        paired.assign(
            event=(paired["target"] == 1).astype(int),
            non_event=(paired["target"] == 0).astype(int),
        )
        .groupby("score", sort=False)[["event", "non_event"]]
        .sum()
        .sort_index(ascending=False)
    )
    cumulative_event = distribution["event"].cumsum() / event_count
    cumulative_non_event = distribution["non_event"].cumsum() / non_event_count
    return float((cumulative_event - cumulative_non_event).abs().max())


def ks(
    score_column: str = "score",
    dataframe: pd.DataFrame | None = None,
    group_columns: str | Sequence[str] | None = None,
    target: str = "target",
) -> pd.DataFrame:
    """Calculate exact KS overall or independently for each Pandas group.

    Pass your data as ``ks(dataframe=df)``. Column names default to ``score``
    and ``target``; ``group_columns=None`` calculates one overall result.
    Supply a column name or sequence of names for per-group results.
    The result contains ``KS`` (between 0 and 1) and any group columns.
    Missing pairs are ignored; KS is NaN when either binary class is absent.
    Existing positional calls remain supported.
    """
    if dataframe is None:
        raise ValueError("dataframe is required; use ks(dataframe=df)")
    groups = (
        [group_columns] if isinstance(group_columns, str) else list(group_columns or [])
    )
    missing = [
        name
        for name in [score_column, target, *groups]
        if name not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not group_columns:
        value = calculate_ks(dataframe[target], dataframe[score_column])
        return pd.DataFrame({"KS": [value]})

    records = []
    for key, group in dataframe.groupby(groups, dropna=False, sort=False):
        keys = key if isinstance(key, tuple) else (key,)
        record = dict(zip(groups, keys, strict=True))
        record["KS"] = calculate_ks(group[target], group[score_column])
        records.append(record)

    return pd.DataFrame.from_records(records, columns=[*groups, "KS"])


def _calculate_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    """Compute binary ROC AUC after removing missing values pairwise.

    Higher scores must indicate the positive class. Return NaN for empty
    data or a single target class; invalid inputs raise scikit-learn errors.
    Series are aligned by index before removing missing pairs.
    """
    data = pd.DataFrame({"target": y_true, "score": y_score}).dropna()
    if data.empty or data["target"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(data["target"], data["score"]))


def _calculate_gini(auc: float) -> float:
    """Compute Gini as ``2 * AUC - 1``, preserving NaN."""
    if pd.isna(auc):
        return float("nan")
    return float(2.0 * auc - 1.0)


def _calculate_auc_gini(y_true: pd.Series, y_score: pd.Series) -> tuple[float, float]:
    """Return ROC AUC and its corresponding Gini coefficient."""
    auc = _calculate_auc(y_true, y_score)
    return auc, _calculate_gini(auc)


def auc_gini(
    score_column: str = "score",
    df: pd.DataFrame | None = None,
    group_by: str | list[str] | None = None,
    target_column: str = "target",
) -> pd.DataFrame:
    """Compute binary ROC AUC and Gini globally or by group.

    Parameters
    ----------
    score_column:
        Score column, default ``score``. Higher scores indicate the positive class.
    df:
        Input DataFrame; supply your data with ``auc_gini(df=df)``.
    group_by:
        Group column or nonempty list of columns. None calculates overall metrics.
        Missing group keys are retained; unobserved categorical groups are omitted.
    target_column:
        Binary target column, default ``target``.

    Returns
    -------
    pandas.DataFrame
        Lowercase ``auc`` and ``gini`` columns, preceded by any grouping columns.
        Missing pairs are removed. Empty data and single-class groups yield NaN.
        An empty grouped input returns an empty result with these columns.

    Raises
    ------
    TypeError
        If df is not a DataFrame.
    KeyError
        If required columns are missing.
    ValueError
        If group_by is invalid or scores/targets are unsuitable for binary AUC.

    Examples
    --------
    >>> auc_gini(df=df, target_column="default_flag")  # doctest: +SKIP
    >>> auc_gini(df=df, group_by="segment")  # doctest: +SKIP
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")

    if group_by is None:
        group_columns = []
    elif isinstance(group_by, str):
        group_columns = [group_by]
    elif isinstance(group_by, list) and all(
        isinstance(column, str) for column in group_by
    ):
        if not group_by:
            raise ValueError("`group_by` cannot be an empty list.")
        group_columns = group_by
    else:
        raise ValueError("`group_by` must be a string, a list of strings, or None.")

    if len(set(group_columns)) != len(group_columns):
        raise ValueError("`group_by` cannot contain duplicate columns.")
    if set(group_columns) & {"auc", "gini"}:
        raise ValueError("Grouping columns cannot be named 'auc' or 'gini'.")
    required_columns = {score_column, target_column, *group_columns}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise KeyError("Missing required column(s): " + ", ".join(missing_columns))

    if group_by is None:
        auc, gini = _calculate_auc_gini(df[target_column], df[score_column])
        return pd.DataFrame({"auc": [auc], "gini": [gini]})

    records = []
    for keys, group in df.groupby(
        group_columns, dropna=False, sort=False, observed=True
    ):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_columns, keys, strict=True))
        record["auc"], record["gini"] = _calculate_auc_gini(
            group[target_column], group[score_column]
        )
        records.append(record)
    return pd.DataFrame.from_records(records, columns=[*group_columns, "auc", "gini"])


def curve_roc(
    score_column: str = "score",
    df: pd.DataFrame | None = None,
    target_column: str = "target",
) -> pd.DataFrame:
    """Return exact ROC points as a DataFrame for plotting.

    Use ``curve_roc(df=df)`` with numeric ``score`` and binary 0/1 ``target``
    columns. Higher scores indicate class 1. Null/NaN pairs are removed.
    Returns ``fpr``, ``tpr``, and ``threshold`` in descending threshold order,
    including the origin and endpoint. The origin threshold is infinity.
    Empty or single-class data return an empty DataFrame with these columns.
    Filter your DataFrame first to plot a particular group.
    Invalid DataFrames, missing columns, and nonbinary targets raise errors.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    missing = sorted({score_column, target_column}.difference(df.columns))
    if missing:
        raise KeyError("Missing required column(s): " + ", ".join(missing))
    target = df[target_column].dropna()
    if not target.isin([0, 1]).all():
        raise ValueError(f"`{target_column}` must contain only binary values 0 and 1.")
    data = df[[score_column, target_column]].dropna()
    if data.empty or data[target_column].nunique() < 2:
        return pd.DataFrame(
            {name: pd.Series(dtype=float) for name in ["fpr", "tpr", "threshold"]}
        )
    fpr, tpr, thresholds = roc_curve(
        data[target_column], data[score_column], pos_label=1, drop_intermediate=False
    )
    thresholds = thresholds.astype(float)
    thresholds[0] = float("inf")
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def risk_table(
    score_column: str = "score",
    df: pd.DataFrame | None = None,
    target_column: str = "target",
    n_tiles: int = 10,
    *,
    ascending: bool = False,
) -> pd.DataFrame:
    """Summarize risk in approximately equal-volume score tiles.

    Use ``risk_table(df=df, n_tiles=10)``. Tile 1 contains the highest scores;
    set ``ascending=True`` when lower scores indicate greater risk.
    Targets must be numeric 0/1. Null/NaN score-target pairs are excluded.
    Scores must be finite numeric values. No input data is modified.

    Returns ``n_tile``, ``minimum_range``, ``maximum_range``, ``total_volume``,
    ``total_events``, ``total_non_events``, and ``event_rate`` (events / volume).
    Bounds are inclusive observed score minima/maxima, not reusable cutoffs:
    tied scores may span tiles. Pandas retains input order within ties.
    Remainder rows go to earlier tiles, matching Spark ntile sizes.
    Only occupied tiles are returned; empty data produce an empty table.
    Filter the input first to summarize a particular segment.
    This function does not calculate KS, AUC, or ROC curves.
    """
    from ._risk import RISK_COLUMNS, validate_risk_options

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    validate_risk_options(
        list(df.columns), score_column, target_column, n_tiles, ascending
    )
    if not pd.api.types.is_numeric_dtype(
        df[score_column]
    ) or pd.api.types.is_complex_dtype(df[score_column]):
        raise TypeError("Scores must be real numeric values.")
    if not df[target_column].dropna().isin([0, 1]).all():
        raise ValueError(f"`{target_column}` must contain only binary values 0 and 1.")
    if df[score_column].isin([float("inf"), float("-inf")]).any():
        raise ValueError("Scores must be finite.")
    data = pd.DataFrame(
        {"_score": df[score_column], "_target": df[target_column]}
    ).dropna()
    data = data.sort_values("_score", ascending=ascending, kind="stable").copy()
    if data.empty:
        return pd.DataFrame(
            {
                name: pd.Series(
                    dtype="float64"
                    if name in {"minimum_range", "maximum_range", "event_rate"}
                    else "int64"
                )
                for name in RISK_COLUMNS
            }
        )
    base, extra = divmod(len(data), n_tiles)
    data["_tile"] = [
        tile
        for tile in range(1, min(n_tiles, len(data)) + 1)
        for _ in range(base + (tile <= extra))
    ]
    data["_target"] = data["_target"].astype("int64")
    result = (
        data.groupby("_tile", sort=True)
        .agg(
            minimum_range=("_score", "min"),
            maximum_range=("_score", "max"),
            total_volume=("_target", "size"),
            total_events=("_target", "sum"),
        )
        .rename_axis("n_tile")
        .reset_index()
    )
    result["total_non_events"] = result["total_volume"] - result["total_events"]
    result["event_rate"] = result["total_events"] / result["total_volume"]
    return result[list(RISK_COLUMNS)]


# Compatibility name for existing notebooks.
roc_auc_gini = auc_gini


__all__ = ["auc_gini", "calculate_ks", "curve_roc", "ks", "risk_table", "roc_auc_gini"]
