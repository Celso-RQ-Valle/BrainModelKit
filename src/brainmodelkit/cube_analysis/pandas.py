"""Exact metric cubes using the internal Pandas metrics."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from brainmodelkit.metrics._risk import RISK_COLUMNS
from brainmodelkit.metrics.pandas import risk_table

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
    ``General``; observed keys become nullable strings. Metrics are rounded to
    five decimals. Output order is group_columns + [KS, AUC, Gini, score]. Missing
    pairs and undefined metrics follow the internal metric functions.
    Threshold counts and cumulative sums are vectorized across all groups and
    scores. Expansion requires O(rows * scores * 2 ** dimensions) memory.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    groups, score_names, subsets = cube_options(
        df.columns, group_columns, score_columns, target_column
    )
    if not df[target_column].dropna().isin([0, 1]).all():
        raise ValueError("Targets must contain only binary values 0 and 1.")
    # Validate scores in one array operation. As in scalar AUC, finite-score
    # validation applies only when a score has both target classes present.
    numeric_scores = df[score_names].to_numpy(dtype=float, na_value=np.nan)
    targets = df[target_column].to_numpy(dtype=float, na_value=np.nan)[:, None]
    paired = ~np.isnan(numeric_scores) & ~np.isnan(targets)
    both_classes = (paired & (targets == 0)).any(axis=0) & (
        paired & (targets == 1)
    ).any(axis=0)
    if (np.isinf(numeric_scores) & paired & both_classes).any():
        raise ValueError("Scores must be finite.")
    if df.empty:
        return pd.DataFrame(
            {
                **{name: ["General"] * len(score_names) for name in groups},
                **{name: np.nan for name in ["KS", "AUC", "Gini"]},
                "score": score_names,
            }
        )

    # Integer codes keep real nulls and real 'General' keys separate from rollups.
    codes = (
        np.column_stack([pd.factorize(df[name], sort=False)[0] for name in groups])
        if groups
        else np.empty((len(df), 0), dtype=int)
    )
    included = np.array([[name in subset for name in groups] for subset in subsets])
    expanded = np.where(included[:, None, :], codes[None, :, :], -2)
    keys = list(range(len(groups)))
    base = (
        pd.DataFrame(expanded.reshape(-1, len(groups)))
        if groups
        else pd.DataFrame(index=range(len(df)))
    )
    base["subset"] = np.repeat(np.arange(len(subsets)), len(df))
    base["row"] = np.tile(np.arange(len(df)), len(subsets))
    long = base.merge(pd.DataFrame({"model": np.arange(len(score_names))}), how="cross")
    partition = ["model", "subset", *keys]
    long["id"] = long.groupby(partition, sort=False, dropna=False).ngroup()
    observed = long.drop_duplicates("id").set_index("id")
    scores = df[score_names].to_numpy()
    long["value"] = scores[long["row"], long["model"]]
    long["target"] = df[target_column].to_numpy()[long["row"]]
    valid = long.dropna(subset=["value", "target"])
    counts = (
        valid.assign(p=valid.target.eq(1).astype(int), n=valid.target.eq(0).astype(int))
        .groupby(["id", "value"], sort=True)[["p", "n"]]
        .sum()
    )
    cumulative = counts.groupby(level=0).cumsum()
    totals = counts.groupby(level=0).transform("sum")
    rates = cumulative / totals.replace(0, np.nan)
    ks = (rates.p - rates.n).abs().groupby(level=0).max()
    # Ascending ranks: positives beat all lower negatives and half of ties.
    area = counts.p * (cumulative.n - counts.n / 2)
    sums = counts.groupby(level=0).sum()
    auc = area.groupby(level=0).sum() / (sums.p * sums.n).replace(0, np.nan)
    observed["KS"] = ks
    observed["AUC"] = auc
    observed["Gini"] = 2 * auc - 1
    observed = observed.sort_values(["model", "subset", "row"])
    result = pd.DataFrame(
        {
            name: observed[index]
            .map(
                dict(enumerate(pd.factorize(df[name], sort=False)[1].astype("string")))
            )
            .astype("string")
            .mask(observed[index].eq(-2), "General")
            for index, name in enumerate(groups)
        },
        index=observed.index,
    )
    result[["KS", "AUC", "Gini"]] = (
        observed[["KS", "AUC", "Gini"]].astype(float).round(5)
    )
    result["score"] = np.asarray(score_names)[observed.model]
    return result.reset_index(drop=True)


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
    columns, and score. Omitted dimensions use General; null keys stay missing.
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
                        [values[group] if group in subset else "General"] * len(table),
                        index=table.index,
                        dtype="string",
                    )
                table["score"] = score
                results[score].append(table[[*groups, *RISK_COLUMNS, "score"]])
    return pd.concat(
        [table for score in score_names for table in results[score]], ignore_index=True
    )


__all__ = ["calculate_metrics", "calculate_ntile"]
