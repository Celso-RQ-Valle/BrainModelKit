"""Spark ML statistics; dataset rows never leave Spark."""

import math
from collections.abc import Sequence

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import ChiSquareTest, Correlation
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, NumericType

from ._selection import (
    SelectionResult,
    columns,
    correlation_result,
    integer,
    number,
    ranked,
    resolve_frame,
)
from ._spark_quality import col, numeric, present


def supervised(
    df: DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    *,
    numeric_features: bool = True,
) -> list[str]:
    features = columns(df, feature_cols, target_col)
    if not isinstance(df.schema[target_col].dataType, (NumericType, BooleanType)):
        raise ValueError("Target must be numeric binary 0/1, not string labels")
    if (
        df.filter(~present(df, target_col) | ~col(target_col).isin(0, 1))
        .limit(1)
        .count()
    ):
        raise ValueError("Target must contain non-null binary 0/1 values")
    if df.select(col(target_col)).distinct().count() != 2:
        raise ValueError("Target must contain both binary classes")
    if numeric_features:
        numeric(df, features)
    return features


def vector_frame(
    df: DataFrame, features: list[str], target_col: str | None = None
) -> DataFrame:
    # Isolate native ML column names from user columns and avoid dot-name ambiguity.
    frame = df.select(
        *[col(f).cast("double").alias(f"x{i}") for i, f in enumerate(features)],
        *([col(target_col).cast("double").alias("label")] if target_col else []),
    )
    return VectorAssembler(
        inputCols=[f"x{i}" for i in range(len(features))], outputCol="features"
    ).transform(frame)


def correlation_filter(
    train_df: DataFrame | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: DataFrame | None = None,
    threshold: float = 0.9,
    method: str = "pearson",
    max_features: int = 256,
) -> SelectionResult:
    """Run native Spark Pearson or Spearman correlation filtering.

    See [Correlation](../../docs/feature_selection.md#correlation) for the
    driver-memory limit and examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = columns(train_df, feature_cols)
    number(threshold, "threshold", maximum=1)
    integer(max_features, "max_features")
    if len(features) > max_features:
        raise ValueError(
            "Correlation exceeds max_features; prefilter or explicitly "
            "raise the matrix budget"
        )
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be pearson or spearman")
    numeric(train_df, features)
    if train_df.limit(2).count() < 2:
        raise ValueError("Correlation requires at least two rows")
    frame = vector_frame(train_df, features)
    matrix = Correlation.corr(frame, "features", method).first()[0].toArray()
    return correlation_result(features, matrix, threshold, method)


def chi_square(
    train_df: DataFrame | None = None,
    target_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: DataFrame | None = None,
    max_p_value: float | None = 0.05,
    top_k: int | None = None,
    max_categories: int = 1000,
) -> SelectionResult:
    """Run Spark categorical independence tests on integer category codes.

    Unlike sklearn count-based chi2, Spark treats each distinct code as a category.

    See [Chi-Square](../../docs/feature_selection.md#chi-square) for input
    semantics, limits, and examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = supervised(train_df, target_col, feature_cols)
    integer(max_categories, "max_categories", 2)
    if max_p_value is not None:
        number(max_p_value, "max_p_value", maximum=1)
    counts = train_df.agg(
        *[F.countDistinct(col(f)).alias(f"v{i}") for i, f in enumerate(features)]
    ).first()
    if any(v > max_categories for v in counts):
        raise ValueError(
            "Chi-Square feature exceeds max_categories; bin/encode explicitly"
        )
    for f in features:
        if train_df.filter((col(f) < 0) | (col(f) != F.floor(col(f)))).limit(1).count():
            raise ValueError("Chi-Square requires nonnegative integer category codes")
    summary = ChiSquareTest.test(
        vector_frame(train_df, features, target_col), "features", "label"
    ).first()
    rows = [
        {
            "feature": f,
            "statistic": float(summary.statistics[i]),
            "p_value": float(summary.pValues[i]),
        }
        for i, f in enumerate(features)
    ]
    result = ranked(rows, "statistic", "chi_square", top_k=top_k)
    for i, row in enumerate(rows):
        row["selected"] = bool(
            row["selected"]
            and counts[i] > 1
            and math.isfinite(row["p_value"])
            and (max_p_value is None or row["p_value"] <= max_p_value)
        )
    result.metadata.update(max_p_value=max_p_value, input_semantics="categorical_codes")
    return result
