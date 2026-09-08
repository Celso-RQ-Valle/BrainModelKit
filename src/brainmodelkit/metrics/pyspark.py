"""PySpark model metrics implemented with native Spark expressions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def _normalize_groups(
    group_columns: str | Sequence[str] | None,
) -> list[str]:
    if not group_columns:
        return []
    if isinstance(group_columns, str):
        return [group_columns]
    return list(group_columns)


def ks_ntile(
    score_column: str = "score",
    dataframe: DataFrame | None = None,
    group_columns: str | Sequence[str] | None = None,
    target: str = "target",
    n_tiles: int = 10,
) -> DataFrame:
    """Calculate binned KS overall or per group using only Spark operations.

    Scores are sorted from highest to lowest within each optional group and
    divided into ``n_tiles`` approximately equal-sized buckets. The returned
    ``KS`` is the maximum absolute difference between cumulative event and
    non-event distributions across those buckets.

    Use ``ks_ntile(dataframe=df)`` with ``score`` and ``target`` columns.
    By default, calculate one overall result with 10 tiles. Set
    ``group_columns`` to a name or sequence of names for grouped results.
    Missing scores and non-binary targets are ignored. KS is NaN when
    either target class is absent. Existing positional calls are supported.
    """
    if isinstance(n_tiles, bool) or not isinstance(n_tiles, int) or n_tiles < 2:
        raise ValueError("n_tiles must be an integer of at least 2")

    if dataframe is None:
        raise ValueError("dataframe is required; use ks_ntile(dataframe=df)")
    groups = _normalize_groups(group_columns)
    missing = [
        name
        for name in [score_column, target, *groups]
        if name not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    try:
        import pyspark.sql.functions as F
        from pyspark.sql import Window
    except ImportError as error:
        raise ImportError(
            "PySpark is required. Install it with "
            "`pip install 'BrainModelKit[pyspark]'`."
        ) from error

    selected = dataframe.select(
        *[F.col(column) for column in groups],
        F.col(score_column).alias("__bmk_score"),
        F.col(target).alias("__bmk_target"),
    ).where(F.col("__bmk_score").isNotNull() & F.col("__bmk_target").isin(0, 1))

    rank_window = Window.partitionBy(*groups).orderBy(F.desc("__bmk_score"))
    ranked = selected.withColumn(
        "__bmk_tile",
        F.ntile(n_tiles).over(rank_window),
    )
    tiles = ranked.groupBy(*groups, "__bmk_tile").agg(
        F.sum(F.when(F.col("__bmk_target") == 1, 1).otherwise(0)).alias("__bmk_events"),
        F.sum(F.when(F.col("__bmk_target") == 0, 1).otherwise(0)).alias(
            "__bmk_non_events"
        ),
    )

    total_window = Window.partitionBy(*groups)
    cumulative_window = (
        Window.partitionBy(*groups)
        .orderBy("__bmk_tile")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    distribution = (
        tiles.withColumn(
            "__bmk_total_events",
            F.sum("__bmk_events").over(total_window),
        )
        .withColumn(
            "__bmk_total_non_events",
            F.sum("__bmk_non_events").over(total_window),
        )
        .withColumn(
            "__bmk_cumulative_events",
            F.sum("__bmk_events").over(cumulative_window),
        )
        .withColumn(
            "__bmk_cumulative_non_events",
            F.sum("__bmk_non_events").over(cumulative_window),
        )
        .withColumn(
            "__bmk_ks",
            F.when(
                (F.col("__bmk_total_events") > 0)
                & (F.col("__bmk_total_non_events") > 0),
                F.abs(
                    F.col("__bmk_cumulative_events") / F.col("__bmk_total_events")
                    - F.col("__bmk_cumulative_non_events")
                    / F.col("__bmk_total_non_events")
                ),
            ),
        )
    )

    if groups:
        return distribution.groupBy(*groups).agg(
            F.coalesce(F.max("__bmk_ks"), F.lit(float("nan"))).alias("KS")
        )
    return distribution.agg(
        F.coalesce(F.max("__bmk_ks"), F.lit(float("nan"))).alias("KS")
    )


def _validate_binary_target(df: DataFrame, target_column: str) -> None:
    """Reject nonmissing target values other than zero and one."""
    from pyspark.sql import functions as F

    target = df[target_column].cast("double")
    invalid = (
        df.filter(df[target_column].isNotNull() & ~F.isnan(target) & ~target.isin(0, 1))
        .limit(1)
        .count()
    )
    if invalid:
        raise ValueError(f"`{target_column}` must contain only binary values 0 and 1.")


def _prepare_auc_data(
    df: DataFrame, score_column: str, target_column: str
) -> DataFrame:
    """Cast to doubles and remove null and NaN target/score pairs."""
    from pyspark.sql import functions as F

    return df.select(
        df[target_column].cast("double").alias("_target"),
        df[score_column].cast("double").alias("_score"),
    ).filter(
        F.col("_target").isNotNull()
        & F.col("_score").isNotNull()
        & ~F.isnan("_target")
        & ~F.isnan("_score")
    )


def _create_score_tiles(df: DataFrame, n_tiles: int) -> DataFrame:
    """Assign descending score quantile tiles."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    return df.withColumn(
        "_tile", F.ntile(n_tiles).over(Window.orderBy(F.desc("_score")))
    )


def _aggregate_roc_tiles(df: DataFrame) -> DataFrame:
    """Count positives and negatives in each tile."""
    from pyspark.sql import functions as F

    return df.groupBy("_tile").agg(
        F.sum(F.when(F.col("_target") == 1, 1).otherwise(0))
        .cast("double")
        .alias("_positive"),
        F.sum(F.when(F.col("_target") == 0, 1).otherwise(0))
        .cast("double")
        .alias("_negative"),
    )


def _calculate_roc_curve(df: DataFrame) -> DataFrame:
    """Compute cumulative ROC rates, guarding undefined single-class rates."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    cumulative = Window.orderBy("_tile").rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )
    total = Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    return (
        df.withColumn("_total_positive", F.sum("_positive").over(total))
        .withColumn("_total_negative", F.sum("_negative").over(total))
        .withColumn("_cum_positive", F.sum("_positive").over(cumulative))
        .withColumn("_cum_negative", F.sum("_negative").over(cumulative))
        .withColumn(
            "_tpr",
            F.when(
                F.col("_total_positive") > 0,
                F.col("_cum_positive") / F.col("_total_positive"),
            ),
        )
        .withColumn(
            "_fpr",
            F.when(
                F.col("_total_negative") > 0,
                F.col("_cum_negative") / F.col("_total_negative"),
            ),
        )
    )


def _calculate_auc_from_roc(roc_df: DataFrame) -> float:
    """Integrate ROC trapezoids, collecting only the final scalar result."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    window = Window.orderBy("_tile")
    row = (
        roc_df.withColumn(
            "_previous_tpr", F.coalesce(F.lag("_tpr").over(window), F.lit(0.0))
        )
        .withColumn("_previous_fpr", F.coalesce(F.lag("_fpr").over(window), F.lit(0.0)))
        .withColumn(
            "_area",
            (F.col("_fpr") - F.col("_previous_fpr"))
            * (F.col("_tpr") + F.col("_previous_tpr"))
            / 2.0,
        )
        .agg(F.sum("_area").alias("auc"))
        .first()
    )
    return float(row["auc"]) if row and row["auc"] is not None else float("nan")


def _calculate_auc(
    df: DataFrame, score_column: str, target_column: str, n_tiles: int
) -> float:
    """Approximate AUC with quantile tiles without collecting input rows."""
    prepared = _prepare_auc_data(df, score_column, target_column)
    tiled = _create_score_tiles(prepared, n_tiles)
    return _calculate_auc_from_roc(_calculate_roc_curve(_aggregate_roc_tiles(tiled)))


def _calculate_gini(auc: float) -> float:
    """Return ``2 * AUC - 1``, naturally preserving NaN."""
    return float(2.0 * auc - 1.0)


def _calculate_auc_gini(
    df: DataFrame, score_column: str, target_column: str, n_tiles: int
) -> tuple[float, float]:
    """Return tile-based ROC AUC and Gini."""
    auc = _calculate_auc(df, score_column, target_column, n_tiles)
    return auc, _calculate_gini(auc)


def auc_gini(
    score_column: str = "score",
    df: DataFrame | None = None,
    group_by: str | list[str] | None = None,
    target_column: str = "target",
    n_tiles: int = 10,
) -> DataFrame:
    """Approximate binary ROC AUC and Gini overall or by Spark group.

    Parameters
    ----------
    score_column:
        Numeric score column, default ``score``. Higher scores indicate class 1.
    df:
        Input Spark DataFrame. Use ``auc_gini(df=df)`` for the default columns.
    group_by:
        Group column or nonempty list of columns; None means overall metrics.
    target_column:
        Numeric binary label column, default ``target``. Only 0 and 1 are valid.
    n_tiles:
        Number of descending score tiles, default 10; must be an integer >= 2.

    Returns
    -------
    pyspark.sql.DataFrame
        Lowercase ``auc`` and ``gini``, preceded by any grouping columns.
        Null/NaN pairs are removed. Empty or single-class data yield NaN.
        Empty grouped input returns an empty result with the expected schema.

    Raises
    ------
    TypeError
        If df is not a Spark DataFrame or n_tiles is not an integer.
    KeyError
        If required columns are missing.
    ValueError
        If grouping, tile count, or binary targets are invalid.

    Notes
    -----
    This is a tile approximation, unlike exact Pandas AUC. Tied scores may
    span tiles and their ordering can affect the result. Overall ranking uses
    an unpartitioned Spark window. Group keys and final metrics are collected
    to the driver, with a separate calculation per group, so use grouping for
    a modest number of groups. Input observations are never collected.
    This function executes Spark jobs immediately.
    """
    try:
        from pyspark.sql import DataFrame
        from pyspark.sql import functions as F
        from pyspark.sql.types import DoubleType, StructField, StructType
    except ImportError as error:
        raise ImportError(
            "PySpark is required. Install it with "
            "`pip install 'BrainModelKit[pyspark]'`."
        ) from error

    if not isinstance(df, DataFrame):
        raise TypeError("`df` must be a PySpark DataFrame.")
    if not isinstance(n_tiles, int) or isinstance(n_tiles, bool):
        raise TypeError("`n_tiles` must be an integer.")
    if n_tiles < 2:
        raise ValueError("`n_tiles` must be greater than or equal to 2.")
    if group_by is None:
        group_columns = []
    elif isinstance(group_by, str):
        group_columns = [group_by]
    elif (
        isinstance(group_by, list)
        and group_by
        and all(isinstance(col, str) for col in group_by)
    ):
        group_columns = group_by
    else:
        raise ValueError(
            "`group_by` must be a string, a non-empty list of strings, or None."
        )
    if len(set(group_columns)) != len(group_columns):
        raise ValueError("`group_by` cannot contain duplicate columns.")
    if set(group_columns) & {"auc", "gini"}:
        raise ValueError("Grouping columns cannot be named 'auc' or 'gini'.")
    missing = sorted(
        {score_column, target_column, *group_columns}.difference(df.columns)
    )
    if missing:
        raise KeyError("Missing required column(s): " + ", ".join(missing))

    _validate_binary_target(df, target_column)
    metric_fields = [
        StructField("auc", DoubleType(), False),
        StructField("gini", DoubleType(), False),
    ]
    if group_by is None:
        metrics = _calculate_auc_gini(df, score_column, target_column, n_tiles)
        return df.sparkSession.createDataFrame([metrics], StructType(metric_fields))

    group_values = (
        df.select(*[df[column] for column in group_columns]).distinct().collect()
    )
    results = []
    for row in group_values:
        condition = F.lit(True)
        for column in group_columns:
            condition = condition & df[column].eqNullSafe(F.lit(row[column]))
        metrics = _calculate_auc_gini(
            df.filter(condition), score_column, target_column, n_tiles
        )
        results.append(tuple(row[column] for column in group_columns) + metrics)
    schema = [
        StructField(df.schema[column].name, df.schema[column].dataType, True)
        for column in group_columns
    ]
    return df.sparkSession.createDataFrame(results, StructType(schema + metric_fields))


__all__ = ["auc_gini", "ks_ntile"]
