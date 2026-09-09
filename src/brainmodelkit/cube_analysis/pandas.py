"""Exact metric cubes using the internal Pandas metrics."""

from collections.abc import Sequence

import pandas as pd

from brainmodelkit.metrics._risk import RISK_COLUMNS
from brainmodelkit.metrics.pandas import auc_gini, ks, risk_table

from ._common import cube_options


def calculate_metrics(
    df: pd.DataFrame,
    grupos: str | Sequence[str] | None = None,
    scores: str | Sequence[str] = ("score",),
    target: str = "target",
) -> pd.DataFrame:
    """Return KS, AUC and Gini for every score and grouping subset.

    Accepts and returns Pandas DataFrames. Omitted dimensions contain
    ``Geral``; observed keys become nullable strings. Metrics are rounded to
    five decimals. Output order is grupos + [KS, AUC, Gini, score]. Missing
    pairs and undefined metrics follow the internal metric functions.
    There are 2 ** len(grupos) calculations per score.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    groups, score_names, subsets = cube_options(df.columns, grupos, scores, target)
    results = []
    for score in score_names:
        for subset in subsets:
            ks_result = ks(score, df, subset or None, target)
            auc_result = auc_gini(score, df, subset or None, target)
            if subset:
                result = auc_result.merge(ks_result, on=subset, how="left")
            else:
                result = auc_result.assign(KS=ks_result["KS"])
            result = result.rename(columns={"auc": "AUC", "gini": "Gini"})
            for group in groups:
                result[group] = (
                    result[group].astype("string") if group in subset else "Geral"
                )
            result["score"] = score
            for metric in ["KS", "AUC", "Gini"]:
                result[metric] = result[metric].astype(float).round(5)
            results.append(result[[*groups, "KS", "AUC", "Gini", "score"]])
    return pd.concat(results, ignore_index=True)


def calculate_ntile(
    df: pd.DataFrame,
    grupos: str | Sequence[str] | None = None,
    scores: str | Sequence[str] = ("score",),
    target: str = "target",
    n_tiles: int = 10,
    *,
    ascending: bool = False,
) -> pd.DataFrame:
    """Build risk tables per score and every grouping subset.

    Tiles are recalculated within each group. Tile 1 has the highest scores
    unless ascending=True. Return grouping keys, the internal risk_table
    columns, and score. Omitted dimensions use Geral; null keys stay missing.
    Only occupied tiles appear; empty input returns an empty cube. Ranges
    and event rates retain the precision of risk_table.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    groups, score_names, subsets = cube_options(df.columns, grupos, scores, target)
    if set(groups) & set(RISK_COLUMNS):
        raise ValueError("Grouping columns conflict with risk table output columns")
    results = []
    for score in score_names:
        # Overall calculation also validates options and data for empty inputs.
        overall = risk_table(score, df, target, n_tiles, ascending=ascending)
        for subset in subsets:
            grouped = (
                df.groupby(subset, dropna=False, sort=False, observed=True)
                if subset
                else [((), df)]
            )
            for keys, data in grouped:
                keys = keys if isinstance(keys, tuple) else (keys,)
                values = dict(zip(subset, keys, strict=True))
                table = (
                    risk_table(score, data, target, n_tiles, ascending=ascending)
                    if subset
                    else overall.copy()
                )
                for group in groups:
                    table[group] = pd.Series(
                        [values[group] if group in subset else "Geral"] * len(table),
                        index=table.index,
                        dtype="string",
                    )
                table["score"] = score
                results.append(table[[*groups, *RISK_COLUMNS, "score"]])
    return pd.concat(results, ignore_index=True)


__all__ = ["calculate_metrics", "calculate_ntile"]
