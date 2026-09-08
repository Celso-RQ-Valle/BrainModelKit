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


__all__ = ["ks_ntile"]
