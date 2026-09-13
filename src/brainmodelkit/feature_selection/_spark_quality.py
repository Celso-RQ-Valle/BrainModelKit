"""Distributed Spark quality filters; only per-feature summaries reach the driver."""

from collections.abc import Sequence
from functools import reduce

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType, NumericType

from ._selection import SelectionResult, columns, groups, integer, number, resolve_frame


def col(name: str) -> Column:
    return F.col("`" + name.replace("`", "``") + "`")


def present(df: DataFrame, name: str) -> Column:
    value = col(name).isNotNull()
    if isinstance(df.schema[name].dataType, (FloatType, DoubleType)):
        value = value & ~F.isnan(col(name))
    return value


def numeric(df: DataFrame, features: list[str], *, allow_missing: bool = False) -> None:
    for f in features:
        if not isinstance(df.schema[f].dataType, NumericType):
            raise ValueError(
                f"Feature {f!r} must be numeric; encode categories explicitly"
            )
    invalid = [
        col(f).cast("double").isin(float("inf"), -float("inf"))
        | (~present(df, f) if not allow_missing else F.lit(False))
        for f in features
    ]
    if df.filter(reduce(lambda a, b: a | b, invalid)).limit(1).count():
        raise ValueError("Features must be finite; impute missing values first")


def completeness(
    train_df: DataFrame | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: DataFrame | None = None,
    min_completeness: float = 0.7,
    group_by: str | Sequence[str] | None = None,
    require_all_groups: bool = False,
) -> SelectionResult:
    """Native global/grouped non-null fractions; floating NaN also counts as missing."""
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = columns(train_df, feature_cols)
    names = groups(train_df, group_by)
    number(min_completeness, "min_completeness", maximum=1)
    if require_all_groups and not names:
        raise ValueError("require_all_groups requires group_by")
    expressions = [
        F.avg(present(train_df, f).cast("double")).alias(f"v{i}")
        for i, f in enumerate(features)
    ]
    summary = train_df.agg(*expressions).first()
    if summary[0] is None:
        raise ValueError("DataFrame must contain at least one row")
    rows = [
        {
            "feature": f,
            "completeness": float(summary[i]),
            "missing_rate": 1 - float(summary[i]),
            "selected": summary[i] >= min_completeness,
        }
        for i, f in enumerate(features)
    ]
    grouped = None
    if names:
        # Aggregate each feature together; explode only the group-level summary.
        summaries = train_df.groupBy(*[col(n) for n in names]).agg(
            *[
                F.avg(present(train_df, f).cast("double")).alias(f"__bmk_{i}")
                for i, f in enumerate(features)
            ]
        )
        # Keep group names separate from aggregation aliases, including unusual names.
        if any(n.startswith("__bmk_") for n in names):
            raise ValueError("group_by names beginning __bmk_ are reserved")
        grouped = summaries.select(
            *[col(n) for n in names],
            F.explode(
                F.array(
                    *[
                        F.struct(
                            F.lit(f).alias("feature"),
                            col(f"__bmk_{i}").alias("completeness"),
                        )
                        for i, f in enumerate(features)
                    ]
                )
            ).alias("__bmk_row"),
        ).select(*[col(n) for n in names], "__bmk_row.*")
        grouped = grouped.withColumn(
            "missing_rate", 1 - col("completeness")
        ).withColumn("selected", col("completeness") >= min_completeness)
        if require_all_groups:
            worst = {
                r["feature"]: r["minimum"]
                for r in grouped.groupBy("feature")
                .agg(F.min("completeness").alias("minimum"))
                .collect()
            }
            for row in rows:
                row["selected"] = (
                    row["selected"] and worst[row["feature"]] >= min_completeness
                )
    return SelectionResult(
        rows,
        "completeness",
        {
            "min_completeness": min_completeness,
            "require_all_groups": require_all_groups,
        },
        grouped_table=grouped,
    )


def variance_filter(
    train_df: DataFrame | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: DataFrame | None = None,
    min_variance: float = 0.0,
) -> SelectionResult:
    """Native population variance over non-missing observations; strict threshold."""
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = columns(train_df, feature_cols)
    number(min_variance, "min_variance")
    numeric(train_df, features, allow_missing=True)
    summary = train_df.agg(
        F.count(F.lit(1)).alias("n"),
        *[
            F.var_pop(F.when(present(train_df, f), col(f).cast("double"))).alias(
                f"v{i}"
            )
            for i, f in enumerate(features)
        ],
    ).first()
    if summary[0] == 0:
        raise ValueError("DataFrame must contain at least one row")
    rows = []
    for i, f in enumerate(features):
        value = float(summary[i + 1]) if summary[i + 1] is not None else float("nan")
        rows.append({"feature": f, "variance": value, "selected": value > min_variance})
    return SelectionResult(rows, "variance", {"min_variance": min_variance, "ddof": 0})


def cardinality(
    train_df: DataFrame | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: DataFrame | None = None,
    min_unique: int = 2,
    max_unique: int | None = None,
    max_unique_ratio: float | None = None,
) -> SelectionResult:
    """Exact native distinct non-missing counts; ratio denominator includes all rows."""
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = columns(train_df, feature_cols)
    integer(min_unique, "min_unique", 0)
    if max_unique is not None:
        integer(max_unique, "max_unique", min_unique)
    if max_unique_ratio is not None:
        number(max_unique_ratio, "max_unique_ratio", maximum=1)
    summary = train_df.agg(
        F.count(F.lit(1)).alias("n"),
        *[
            F.countDistinct(F.when(present(train_df, f), col(f))).alias(f"v{i}")
            for i, f in enumerate(features)
        ],
    ).first()
    if summary[0] == 0:
        raise ValueError("DataFrame must contain at least one row")
    rows = []
    for i, f in enumerate(features):
        count = int(summary[i + 1])
        ratio = count / summary[0]
        rows.append(
            {
                "feature": f,
                "unique_count": count,
                "unique_ratio": ratio,
                "selected": count >= min_unique
                and (max_unique is None or count <= max_unique)
                and (max_unique_ratio is None or ratio <= max_unique_ratio),
            }
        )
    return SelectionResult(rows, "cardinality")
