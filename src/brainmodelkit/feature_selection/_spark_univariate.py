"""Distributed univariate statistics; only scalar summaries reach Python.

Usage, method algorithms, parameters, and examples:
https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection.md
"""

import math

from pyspark.sql import functions as F

from ._selection import integer, number, ranked, resolve_frame
from ._spark_quality import col
from ._spark_statistics import supervised


def mutual_information(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    threshold=None,
    top_k=None,
    discrete_features=False,
    n_bins=10,
    relative_error=0.001,
    max_categories=1000,
):
    """Estimate empirical binned MI in nats using distributed contingency counts.

    Continuous features use training quantiles; discrete features use exact levels.

    Usage
    -----
    Import `brainmodelkit.feature_selection.pyspark` as `fs`, then call:

        result = fs.mutual_information(
            train_df, "target", ["income", "age"], top_k=1
        )
        print(result.selected_features)

    Parameters, return fields, algorithm, assumptions, and complete examples:
    https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/mutual_information.md
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = supervised(train_df, target_col, feature_cols)
    integer(n_bins, "n_bins", 2)
    integer(max_categories, "max_categories", 2)
    number(relative_error, "relative_error", maximum=1)
    flags = (
        [discrete_features] * len(features)
        if isinstance(discrete_features, bool)
        else list(discrete_features)
    )
    if len(flags) != len(features) or any(type(v) is not bool for v in flags):
        raise ValueError("discrete_features must be a Boolean or per-feature mask")
    n = train_df.count()
    labels = (
        train_df.groupBy(col(target_col).alias("y"))
        .count()
        .withColumnRenamed("count", "ny")
    )
    rows, boundaries = [], {}
    for feature, discrete in zip(features, flags, strict=True):
        edges = []
        value = col(feature)
        if discrete:
            if (
                train_df.select(value).distinct().limit(max_categories + 1).count()
                > max_categories
            ):
                raise ValueError("Discrete feature exceeds max_categories")
        else:
            edges = sorted(
                set(
                    train_df.select(value.alias("v")).approxQuantile(
                        "v", [i / n_bins for i in range(1, n_bins)], relative_error
                    )
                )
            )
            value = F.lit(0)
            for edge in edges:
                value = value + F.when(col(feature) > edge, 1).otherwise(0)
        counts = (
            train_df.select(value.alias("x"), col(target_col).alias("y"))
            .groupBy("x", "y")
            .count()
        )
        marginals = counts.groupBy("x").agg(F.sum("count").alias("nx"))
        score = (
            counts.join(marginals, "x")
            .join(labels, "y")
            .agg(
                F.sum(
                    F.col("count")
                    / n
                    * F.log(F.col("count") * n / (F.col("nx") * F.col("ny")))
                )
            )
            .first()[0]
        )
        rows.append({"feature": feature, "mutual_information": max(0.0, float(score))})
        boundaries[feature] = edges
    result = ranked(rows, "mutual_information", "mutual_information", threshold, top_k)
    result.metadata.update(
        estimator="binned",
        bin_edges=boundaries,
        n_bins=n_bins,
        discrete_features=flags,
        relative_error=relative_error,
    )
    return result


def anova(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    max_p_value=0.05,
    top_k=None,
):
    """One-way F-test from distributed class counts, means, and sample variances.

    Usage
    -----
    Import `brainmodelkit.feature_selection.pyspark` as `fs`, then call:

        result = fs.anova(
            train_df, "target", ["income", "age"], max_p_value=0.05
        )
        print(result.selected_features)

    Parameters, return fields, algorithm, assumptions, and complete examples:
    https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/anova.md
    """
    from scipy.stats import f as f_distribution

    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = supervised(train_df, target_col, feature_cols)
    if max_p_value is not None:
        number(max_p_value, "max_p_value", maximum=1)
    rows = []
    for feature in features:
        summaries = (
            train_df.groupBy(col(target_col))
            .agg(
                F.count("*").alias("n"),
                F.avg(col(feature)).alias("mean"),
                F.var_samp(col(feature)).alias("variance"),
            )
            .collect()
        )  # Exactly two binary-class summaries, never source rows.
        n = sum(r.n for r in summaries)
        if n <= 2:
            raise ValueError("ANOVA requires residual degrees of freedom")
        mean = sum(r.n * r.mean for r in summaries) / n
        between = sum(r.n * (r.mean - mean) ** 2 for r in summaries)
        within = sum((r.n - 1) * (r.variance or 0.0) for r in summaries)
        statistic = (
            between / (within / (n - 2))
            if within > 0
            else float("inf")
            if between > 0
            else float("nan")
        )
        rows.append(
            {
                "feature": feature,
                "f_statistic": statistic,
                "p_value": float(f_distribution.sf(statistic, 1, n - 2)),
            }
        )
    result = ranked(rows, "f_statistic", "anova", top_k=top_k)
    for row in rows:
        row["selected"] = bool(
            row["selected"]
            and math.isfinite(row["p_value"])
            and (max_p_value is None or row["p_value"] <= max_p_value)
        )
    result.metadata["max_p_value"] = max_p_value
    return result
