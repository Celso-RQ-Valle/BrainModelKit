"""Pandas model metrics."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


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


__all__ = ["calculate_ks", "ks"]
