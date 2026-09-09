"""Exact metric cubes using the internal Pandas metrics."""

from collections.abc import Sequence

import pandas as pd

from brainmodelkit.metrics._risk import RISK_COLUMNS
from brainmodelkit.metrics.pandas import _calculate_auc_gini, calculate_ks, risk_table

from ._common import cube_options


def calculate_metrics(
    df: pd.DataFrame,
    group_columns: str | Sequence[str] | None = None,
    score_columns: str | Sequence[str] = ("score",),
    target_column: str = "target",
) -> pd.DataFrame:
    """Return KS, AUC and Gini for every score and grouping subset.

    Use ``df=df``; ``group_columns=None`` (or []) gives overall results.
    ``score_columns`` defaults to ("score",); ``target_column`` to "target".

    Accepts and returns Pandas DataFrames. Omitted dimensions contain
    ``Geral``; observed keys become nullable strings. Metrics are rounded to
    five decimals. Output order is group_columns + [KS, AUC, Gini, score]. Missing
    pairs and undefined metrics follow the internal metric functions.
    There are 2 ** len(group_columns) calculations per score.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    groups, score_names, subsets = cube_options(
        df.columns, group_columns, score_columns, target_column
    )
    # Group once per subset; keep score/subset/group order and concat dtypes.
    results = {score: [] for score in score_names}
    for subset in subsets:
        records = {score: [] for score in score_names}
        grouped = (
            df.groupby(subset, dropna=False, sort=False, observed=True)
            if subset
            else [((), df)]
        )
        for keys, data in grouped:
            keys = keys if isinstance(keys, tuple) else (keys,)
            values = dict(zip(subset, keys, strict=True))
            for score in score_names:
                auc, gini = _calculate_auc_gini(data[target_column], data[score])
                records[score].append(
                    {
                        **values,
                        "KS": calculate_ks(data[target_column], data[score]),
                        "AUC": auc,
                        "Gini": gini,
                    }
                )
        for score in score_names:
            table = pd.DataFrame.from_records(
                records[score], columns=[*subset, "KS", "AUC", "Gini"]
            )
            for group in groups:
                table[group] = (
                    table[group].astype("string") if group in subset else "Geral"
                )
            table["score"] = score
            results[score].append(table[[*groups, "KS", "AUC", "Gini", "score"]])
    result = pd.concat(
        [table for score in score_names for table in results[score]], ignore_index=True
    )
    for metric in ["KS", "AUC", "Gini"]:
        result[metric] = result[metric].astype(float).round(5)
    return result


def calculate_ntile(
    df: pd.DataFrame,
    group_columns: str | Sequence[str] | None = None,
    score_columns: str | Sequence[str] = ("score",),
    target_column: str = "target",
    n_tiles: int = 10,
    *,
    ascending: bool = False,
) -> pd.DataFrame:
    """Build risk tables per score and every grouping subset.

    Use ``df=df``; ``group_columns=None`` (or []) gives overall results.
    ``score_columns`` defaults to ("score",); ``target_column`` to "target".

    Tiles are recalculated within each group. Tile 1 has the highest scores
    unless ascending=True. Return grouping keys, the internal risk_table
    columns, and score. Omitted dimensions use Geral; null keys stay missing.
    Only occupied tiles appear; empty input returns an empty cube. Ranges
    and event rates retain the precision of risk_table.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    groups, score_names, subsets = cube_options(
        df.columns, group_columns, score_columns, target_column
    )
    if set(groups) & set(RISK_COLUMNS):
        raise ValueError("Grouping columns conflict with risk table output columns")
    # Validate each score on the overall table, then reuse each group slice.
    overall = {
        score: risk_table(score, df, target_column, n_tiles, ascending=ascending)
        for score in score_names
    }
    results = {score: [] for score in score_names}
    for subset in subsets:
        grouped = (
            df.groupby(subset, dropna=False, sort=False, observed=True)
            if subset
            else [((), df)]
        )
        for keys, data in grouped:
            keys = keys if isinstance(keys, tuple) else (keys,)
            values = dict(zip(subset, keys, strict=True))
            for score in score_names:
                table = (
                    risk_table(score, data, target_column, n_tiles, ascending=ascending)
                    if subset
                    else overall[score].copy()
                )
                for group in groups:
                    table[group] = pd.Series(
                        [values[group] if group in subset else "Geral"] * len(table),
                        index=table.index,
                        dtype="string",
                    )
                table["score"] = score
                results[score].append(table[[*groups, *RISK_COLUMNS, "score"]])
    return pd.concat(
        [table for score in score_names for table in results[score]], ignore_index=True
    )


__all__ = ["calculate_metrics", "calculate_ntile"]
