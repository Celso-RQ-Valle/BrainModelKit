"""Metric cubes using internal Spark metrics without Pandas conversion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from brainmodelkit.metrics._risk import RISK_COLUMNS, validate_risk_options
from brainmodelkit.metrics.pyspark import _validate_binary_target, ks_ntile

from ._common import cube_options

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def _grouped_tiles(df, score, target, groups, n_tiles, ascending=False):
    """Apply the internal risk/AUC preparation and ntile rules per partition."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    keys = [f"_group_{index}" for index in range(len(groups))]
    selected = df.select(
        *[df[group].alias(key) for group, key in zip(groups, keys, strict=True)],
        df[score].cast("double").alias("_score"),
        df[target].cast("double").alias("_target"),
    )
    prepared = selected.filter(
        F.col("_score").isNotNull()
        & ~F.isnan("_score")
        & F.col("_target").isNotNull()
        & ~F.isnan("_target")
    )
    window = Window.partitionBy(*keys).orderBy(
        F.asc("_score") if ascending else F.desc("_score")
    )
    return selected, prepared.withColumn("_tile", F.ntile(n_tiles).over(window)), keys


def _grouped_auc(df, score, target, groups, n_tiles):
    """Integrate the internal tile ROC formula without per-group driver jobs."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    selected, tiled, keys = _grouped_tiles(df, score, target, groups, n_tiles)
    tiles = tiled.groupBy(*keys, "_tile").agg(
        F.sum(F.when(F.col("_target") == 1, 1).otherwise(0)).cast("double").alias("_p"),
        F.sum(F.when(F.col("_target") == 0, 1).otherwise(0)).cast("double").alias("_n"),
    )
    order = Window.partitionBy(*keys).orderBy("_tile")
    cumulative = order.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    total = Window.partitionBy(*keys)
    roc = (
        tiles.withColumn("_total_p", F.sum("_p").over(total))
        .withColumn("_total_n", F.sum("_n").over(total))
        .withColumn(
            "_tpr",
            F.when(
                F.col("_total_p") > 0, F.sum("_p").over(cumulative) / F.col("_total_p")
            ),
        )
        .withColumn(
            "_fpr",
            F.when(
                F.col("_total_n") > 0, F.sum("_n").over(cumulative) / F.col("_total_n")
            ),
        )
        .withColumn("_prev_tpr", F.coalesce(F.lag("_tpr").over(order), F.lit(0.0)))
        .withColumn("_prev_fpr", F.coalesce(F.lag("_fpr").over(order), F.lit(0.0)))
    )
    result = roc.groupBy(*keys).agg(
        F.sum(
            (F.col("_fpr") - F.col("_prev_fpr"))
            * (F.col("_tpr") + F.col("_prev_tpr"))
            / 2.0
        ).alias("auc")
    )
    if keys:
        # Retain groups whose score/target pairs were all missing.
        observed = selected.select(*keys).distinct().alias("_observed")
        result = result.alias("_auc")
        condition = F.lit(True)
        for key in keys:
            condition = condition & observed[key].eqNullSafe(result[key])
        result = observed.join(result, condition, "left").select(
            *[observed[key] for key in keys], result["auc"]
        )
    return result.select(
        *[result[key].alias(group) for key, group in zip(keys, groups, strict=True)],
        F.coalesce(result["auc"], F.lit(float("nan"))).alias("auc"),
        (2 * F.coalesce(result["auc"], F.lit(float("nan"))) - 1).alias("gini"),
    )


def calculate_metrics(
    df: DataFrame,
    group_columns: str | Sequence[str] | None = None,
    score_columns: str | Sequence[str] = ("score",),
    target_column: str = "target",
    n_tiles: int = 10,
) -> DataFrame:
    """Return a Spark cube of tile-based KS, AUC and Gini.

    Use ``df=df``; ``group_columns=None`` (or []) gives overall results.
    ``score_columns`` defaults to ("score",); ``target_column`` to "target".

    Every grouping subset is computed for each score using internal Spark
    metrics. Omitted dimensions are ``Geral``; observed keys are cast to
    strings, retaining nulls. Metrics are rounded to five decimals.
    No Pandas conversion, Python UDF, or group-key collection is used.
    AUC is calculated across groups with partitioned windows; target validation
    runs once. Subsets still grow as 2 ** len(group_columns).
    """
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F

    if not isinstance(df, DataFrame):
        raise TypeError("df must be a PySpark DataFrame")
    groups, score_names, subsets = cube_options(
        df.columns, group_columns, score_columns, target_column
    )
    if isinstance(n_tiles, bool) or not isinstance(n_tiles, int) or n_tiles < 2:
        raise ValueError("n_tiles must be an integer of at least 2")
    _validate_binary_target(df, target_column)
    result = None
    for score in score_names:
        for subset in subsets:
            ks_result = ks_ntile(score, df, subset or None, target_column, n_tiles)
            auc_result = _grouped_auc(df, score, target_column, subset, n_tiles)
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
    group_columns: str | Sequence[str] | None = None,
    score_columns: str | Sequence[str] = ("score",),
    target_column: str = "target",
    n_tiles: int = 10,
    *,
    ascending: bool = False,
) -> DataFrame:
    """Build a Spark risk-table cube with tiles recalculated per group.

    Use ``df=df``; ``group_columns=None`` (or []) gives overall results.
    ``score_columns`` defaults to ("score",); ``target_column`` to "target".

    Uses the internal risk-table formulas with partitioned Spark operations,
    without Pandas, UDFs, or collecting group keys. Tile 1
    contains highest scores unless ascending=True. Output is grouping keys,
    risk_table columns, and score. Omitted dimensions use Geral, null keys
    remain null. Empty input returns an empty cube. Only occupied tiles appear.
    Ranges and event rates retain risk_table precision. Row order is unspecified.
    """
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F

    if not isinstance(df, DataFrame):
        raise TypeError("df must be a PySpark DataFrame")
    groups, score_names, subsets = cube_options(
        df.columns, group_columns, score_columns, target_column
    )
    if set(groups) & set(RISK_COLUMNS):
        raise ValueError("Grouping columns conflict with risk table output columns")
    from pyspark.sql.types import BooleanType, NumericType

    for score in score_names:
        validate_risk_options(df.columns, score, target_column, n_tiles, ascending)
        if not isinstance(df.schema[score].dataType, NumericType):
            raise TypeError("Scores must be real numeric values.")
    if not isinstance(df.schema[target_column].dataType, (NumericType, BooleanType)):
        raise TypeError("Targets must be numeric binary values.")
    _validate_binary_target(df, target_column)
    infinite = F.lit(False)
    for score in score_names:
        infinite = infinite | df[score].isin(float("inf"), float("-inf"))
    if df.filter(infinite).limit(1).count():
        raise ValueError("Scores must be finite.")
    result = None
    for score in score_names:
        for subset in subsets:
            _, tiled, keys = _grouped_tiles(
                df, score, target_column, subset, n_tiles, ascending
            )
            table = (
                tiled.groupBy(*keys, "_tile")
                .agg(
                    F.min("_score").alias("minimum_range"),
                    F.max("_score").alias("maximum_range"),
                    F.count("_target").alias("total_volume"),
                    F.sum(F.col("_target").cast("long")).alias("total_events"),
                )
                .withColumnRenamed("_tile", "n_tile")
            )
            table = table.withColumn(
                "total_non_events", F.col("total_volume") - F.col("total_events")
            ).withColumn("event_rate", F.col("total_events") / F.col("total_volume"))
            key_map = dict(zip(subset, keys, strict=True))
            current = table.select(
                *[
                    (
                        table[key_map[group]].cast("string")
                        if group in subset
                        else F.lit("Geral")
                    ).alias(group)
                    for group in groups
                ],
                *RISK_COLUMNS,
                F.lit(score).alias("score"),
            )
            result = current if result is None else result.unionByName(current)
    return result


__all__ = ["calculate_metrics", "calculate_ntile"]
