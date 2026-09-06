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
    score_column: str,
    dataframe: pd.DataFrame,
    group_columns: str | Sequence[str] | None,
    target: str,
) -> pd.DataFrame:
    """Calculate exact KS overall or independently for each Pandas group."""
    if not group_columns:
        value = calculate_ks(dataframe[target], dataframe[score_column])
        return pd.DataFrame({"KS": [value]})

    groups = [group_columns] if isinstance(group_columns, str) else list(group_columns)
    records = []
    for key, group in dataframe.groupby(groups, dropna=False, sort=False):
        keys = key if isinstance(key, tuple) else (key,)
        record = dict(zip(groups, keys, strict=True))
        record["KS"] = calculate_ks(group[target], group[score_column])
        records.append(record)

    return pd.DataFrame.from_records(records, columns=[*groups, "KS"])


__all__ = ["calculate_ks", "ks"]
