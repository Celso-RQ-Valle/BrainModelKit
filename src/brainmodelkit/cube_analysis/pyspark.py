"""Metric cubes using internal Spark metrics without Pandas conversion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from brainmodelkit.metrics._risk import RISK_COLUMNS
from brainmodelkit.metrics.pyspark import auc_gini, ks_ntile, risk_table

from ._common import cube_options

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def calculate_metrics(
    df: DataFrame,
    grupos: str | Sequence[str] | None = None,
    scores: str | Sequence[str] = ("score",),
    target: str = "target",
    n_tiles: int = 10,
) -> DataFrame:
    """Return a Spark cube of tile-based KS, AUC and Gini.

    Every grouping subset is computed for each score using internal Spark
    metrics. Omitted dimensions are ``Geral``; observed keys are cast to
    strings, retaining nulls. Metrics are rounded to five decimals.
    No Pandas conversion or Python UDF is used. Internal AUC collects group
    keys and final metrics and executes jobs per group. Use modest numbers
    of dimensions and groups: subsets grow as 2 ** len(grupos).
    """
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F

    if not isinstance(df, DataFrame):
        raise TypeError("df must be a PySpark DataFrame")
    groups, score_names, subsets = cube_options(df.columns, grupos, scores, target)
    result = None
    for score in score_names:
        for subset in subsets:
            ks_result = ks_ntile(score, df, subset or None, target, n_tiles)
            auc_result = auc_gini(score, df, subset or None, target, n_tiles)
            if subset:
                condition = F.lit(True)
                for group in subset:
                    condition = condition & auc_result[group].eqNullSafe(
                        ks_result[group]
                    )
                merged = auc_result.join(ks_result, condition, "left").select(
                    *[auc_result[group] for group in subset],
                    ks_result["KS"],
                    auc_result["auc"],
                    auc_result["gini"],
                )
            else:
                merged = auc_result.crossJoin(ks_result)
            current = merged.select(
                *[
                    (
                        merged[group].cast("string")
                        if group in subset
                        else F.lit("Geral")
                    ).alias(group)
                    for group in groups
                ],
                F.round(F.coalesce(merged["KS"], F.lit(float("nan"))), 5).alias("KS"),
                F.round(merged["auc"], 5).alias("AUC"),
                F.round(merged["gini"], 5).alias("Gini"),
                F.lit(score).alias("score"),
            )
            result = current if result is None else result.unionByName(current)
    return result


def calculate_ntile(
    df: DataFrame,
    grupos: str | Sequence[str] | None = None,
    scores: str | Sequence[str] = ("score",),
    target: str = "target",
    n_tiles: int = 10,
    *,
    ascending: bool = False,
) -> DataFrame:
    """Build a Spark risk-table cube with tiles recalculated per group.

    Reuses risk_table without Pandas or UDFs. Collects distinct group keys,
    but never input observations; use modest group cardinalities. Tile 1
    contains highest scores unless ascending=True. Output is grouping keys,
    risk_table columns, and score. Omitted dimensions use Geral, null keys
    remain null. Empty input returns an empty cube. Only occupied tiles appear.
    Ranges and event rates retain risk_table precision. Row order is unspecified.
    """
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F

    if not isinstance(df, DataFrame):
        raise TypeError("df must be a PySpark DataFrame")
    groups, score_names, subsets = cube_options(df.columns, grupos, scores, target)
    if set(groups) & set(RISK_COLUMNS):
        raise ValueError("Grouping columns conflict with risk table output columns")
    result = None
    for score in score_names:
        overall = risk_table(score, df, target, n_tiles, ascending=ascending)
        for subset in subsets:
            keys = df.select(*subset).distinct().collect() if subset else [{}]
            for values in keys:
                if subset:
                    condition = F.lit(True)
                    for group in subset:
                        condition = condition & df[group].eqNullSafe(
                            F.lit(values[group])
                        )
                    table = risk_table(
                        score,
                        df.filter(condition),
                        target,
                        n_tiles,
                        ascending=ascending,
                    )
                else:
                    table = overall
                current = table.select(
                    *[
                        F.lit(values[group] if group in subset else "Geral")
                        .cast("string")
                        .alias(group)
                        for group in groups
                    ],
                    *RISK_COLUMNS,
                    F.lit(score).alias("score"),
                )
                result = current if result is None else result.unionByName(current)
    return result


__all__ = ["calculate_metrics", "calculate_ntile"]
