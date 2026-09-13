"""Distributed binary Information Value and WoE diagnostics.

Usage, method algorithms, parameters, and examples:
https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection.md
"""

from collections.abc import Sequence
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    NullType,
    NumericType,
    StringType,
    TimestampType,
)

from ._selection import SelectionResult, integer, number, ranked, resolve_frame
from ._spark_quality import col, numeric, present
from ._spark_statistics import supervised


def information_value(
    train_df: DataFrame | None = None,
    target_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: DataFrame | None = None,
    n_bins: int = 10,
    min_iv: float | None = None,
    smoothing: float = 0.5,
    binning: str = "quantile",
    relative_error: float = 0.001,
) -> SelectionResult:
    """Compute native Spark binned and smoothed binary Information Value.

    Usage
    -----
    Import `brainmodelkit.feature_selection.pyspark` as `fs`, then call:

        result = fs.information_value(
            train_df, "target", ["income", "age"], min_iv=0.02
        )
        print(result.selected_features)

    Parameters, return fields, algorithm, assumptions, and complete examples:
    https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/information_value.md
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = supervised(train_df, target_col, feature_cols, numeric_features=False)
    integer(n_bins, "n_bins", 2)
    number(smoothing, "smoothing")
    number(relative_error, "relative_error", maximum=1)
    if smoothing == 0:
        raise ValueError("smoothing must be > 0")
    if binning not in ("quantile", "uniform"):
        raise ValueError("binning must be quantile or uniform")
    rows, tables, edges_by_feature = [], [], {}
    for f in features:
        is_numeric = isinstance(train_df.schema[f].dataType, NumericType)
        if not isinstance(
            train_df.schema[f].dataType,
            (NumericType, BooleanType, DateType, StringType, TimestampType, NullType),
        ):
            raise ValueError("IV requires numeric or scalar categorical features")
        base = train_df.select(
            col(target_col).cast("double").alias("event"),
            F.when(present(train_df, f), col(f)).alias("value"),
        )
        edges = []
        if is_numeric:
            numeric(train_df, [f], allow_missing=True)
            base = base.withColumn("value", col("value").cast("double"))
            if binning == "quantile":
                edges = sorted(
                    set(
                        base.approxQuantile(
                            "value",
                            [i / n_bins for i in range(1, n_bins)],
                            relative_error,
                        )
                    )
                )
            else:
                bounds = base.agg(F.min("value"), F.max("value")).first()
                if bounds[0] is not None and bounds[0] < bounds[1]:
                    edges = [
                        bounds[0] + (bounds[1] - bounds[0]) * i / n_bins
                        for i in range(1, n_bins)
                    ]
            # Count cut points strictly below each value. Equality belongs to
            # the lower bin, including a quantile cut point at the minimum.
            code = F.lit(0)
            for edge in edges:
                code = code + F.when(col("value") > edge, 1).otherwise(0)
            binned = base.select(
                "event",
                F.when(col("value").isNotNull(), code.cast("string")).alias("bin"),
            )
        else:
            binned = base.select("event", col("value").cast("string").alias("bin"))
        edges_by_feature[f] = edges
        counts = binned.groupBy("bin").agg(
            F.sum("event").alias("event_count"),
            (F.count(F.lit(1)) - F.sum("event")).alias("non_event_count"),
        )
        # A single aggregate row, independent of dataset or category cardinality.
        totals = counts.agg(
            F.sum("event_count"), F.sum("non_event_count"), F.count(F.lit(1))
        ).first()
        table = counts.withColumn(
            "distribution_event",
            (col("event_count") + smoothing) / (totals[0] + smoothing * totals[2]),
        )
        table = table.withColumn(
            "distribution_non_event",
            (col("non_event_count") + smoothing) / (totals[1] + smoothing * totals[2]),
        )
        table = table.withColumn(
            "woe", F.log(col("distribution_event") / col("distribution_non_event"))
        )
        table = table.withColumn(
            "iv_component",
            (col("distribution_event") - col("distribution_non_event")) * col("woe"),
        )
        table = table.withColumn("feature", F.lit(f)).withColumn(
            "is_missing", col("bin").isNull()
        )
        value = float(table.agg(F.sum("iv_component")).first()[0])
        rows.append({"feature": f, "iv": value})
        tables.append(table)
    result = ranked(rows, "iv", "information_value", min_iv)
    result.bin_details = reduce(lambda a, b: a.unionByName(b), tables)
    result.metadata.update(
        n_bins=n_bins,
        binning=binning,
        smoothing=smoothing,
        bin_edges=edges_by_feature,
        relative_error=relative_error,
    )
    return result
