"""Distributed bureau simulation and point-in-time features."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from datetime import date
from functools import reduce
from typing import TYPE_CHECKING

from brainmodelkit._temporal import (
    AGGREGATIONS,
    build_config,
    columns,
    feature_config,
    simulate_entity,
    simulation_config,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def _col(name):
    from pyspark.sql import functions as F

    return F.col("`" + name.replace("`", "``") + "`")


def _validate_dates(frame, keys, day):
    from pyspark.sql import functions as F

    dtype = dict(frame.dtypes)[day]
    if dtype not in ("date", "timestamp", "timestamp_ntz", "string"):
        raise ValueError("Dates must be date, timestamp, or ISO date strings")
    value = _col(day)
    quoted = "`" + day.replace("`", "``") + "`"
    parsed = F.expr(f"try_cast({quoted} as date)")
    timestamp = F.expr(f"try_cast({quoted} as timestamp)")
    invalid = value.isNull() | parsed.isNull() | timestamp.isNull()
    invalid = invalid | (timestamp != parsed.cast("timestamp"))
    if dtype == "string":
        invalid = invalid | ~value.rlike(r"^\d{4}-\d{2}-\d{2}$")
    for key in keys:
        invalid = invalid | _col(key).isNull()
    if frame.where(invalid).limit(1).count():
        raise ValueError("Keys/dates must be non-null; dates must be at midnight")


def simulate_bureau_history(
    customers: DataFrame,
    entity_cols: str | Sequence[str],
    initial_date: str | date,
    final_date: str | date,
    frequency: str = "monthly",
    seed: int = 42,
    *,
    late_entry_fraction: float = 0.1,
    missing_customer_fraction: float = 0.1,
    missing_period_probability: float = 0.15,
    score_min: float = 0,
    score_max: float = 1000,
    persistence: float = 0.85,
    noise_std: float = 25,
    observation_col: str = "observation_date",
    score_col: str = "bureau_score",
) -> DataFrame:
    """Distributed equivalent of pandas.simulate_bureau_history.

    A worker-side UDF generates one bounded period array per distinct customer;
    no customer/history data is collected. See docs/temporal.md for alignment.
    """
    from pyspark.sql import functions as F

    keys, dates = simulation_config(
        entity_cols,
        initial_date,
        final_date,
        frequency,
        seed,
        late_entry_fraction,
        missing_customer_fraction,
        missing_period_probability,
        score_min,
        score_max,
        persistence,
        noise_std,
        observation_col,
        score_col,
    )
    columns(customers, keys)
    if (
        customers.where(reduce(lambda a, b: a | b, [_col(k).isNull() for k in keys]))
        .limit(1)
        .count()
    ):
        raise ValueError("Customer keys must not be null")

    @F.udf("array<struct<day:date,score:double>>")
    def path(*key):
        return simulate_entity(
            key,
            dates,
            seed,
            late_entry_fraction,
            missing_customer_fraction,
            missing_period_probability,
            score_min,
            score_max,
            persistence,
            noise_std,
        )

    internal = [f"k{i}" for i in range(len(keys))]
    base = customers.select(
        *[_col(k).alias(n) for k, n in zip(keys, internal, strict=True)]
    ).distinct()
    return base.select("*", F.explode(path(*map(_col, internal))).alias("p")).select(
        *[_col(n).alias(k) for k, n in zip(keys, internal, strict=True)],
        F.col("p.day").alias(observation_col),
        F.col("p.score").alias(score_col),
    )


def temporal_features(
    spine: DataFrame,
    history: DataFrame,
    entity_cols: str | Sequence[str],
    score_cols: str | Sequence[str],
    *,
    reference_col: str = "dt_ref",
    observation_col: str = "observation_date",
    unit: str = "months",
    windows: Sequence[int] = (3, 6, 9),
    lags: Sequence[int] = (1, 2, 8),
    aggregations: Sequence[str] = AGGREGATIONS,
    max_staleness_days: int | None = None,
    _date_diff: bool = False,
) -> DataFrame:
    """Distributed entity-keyed as-of features; preserves all spine rows.

    Windows are (reference - k periods, reference]. Lookup dates accompany each
    latest non-null score. Month arithmetic is calendar-based. Duplicate history
    keys/dates fail. Sample std is null for fewer than two non-null observations.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import NumericType

    keys, scores, windows, lags, aggs, output = feature_config(
        entity_cols,
        score_cols,
        reference_col,
        observation_col,
        unit,
        windows,
        lags,
        aggregations,
        max_staleness_days,
        date_diff=_date_diff,
    )
    columns(spine, [*keys, reference_col])
    columns(history, [*keys, observation_col, *scores])
    if set(output) & set(spine.columns):
        raise ValueError("Feature columns already exist in spine")
    for key in keys:
        if spine.schema[key].dataType != history.schema[key].dataType:
            raise ValueError("Entity key types must match between spine and history")
    if any(not isinstance(history.schema[s].dataType, NumericType) for s in scores):
        raise ValueError("Value columns must be numeric")
    _validate_dates(spine, keys, reference_col)
    _validate_dates(history, keys, observation_col)
    internal = [f"k{i}" for i in range(len(keys))]
    refs = spine.select(
        *[_col(k).alias(n) for k, n in zip(keys, internal, strict=True)],
        _col(reference_col).cast("date").alias("r"),
    ).distinct()
    hist = history.select(
        *[_col(k).alias("h" + n) for k, n in zip(keys, internal, strict=True)],
        _col(observation_col).cast("date").alias("d"),
        *[
            F.when(~F.isnan(_col(s)), _col(s).cast("double")).alias(f"v{i}")
            for i, s in enumerate(scores)
        ],
    )
    if (
        hist.groupBy(*["h" + n for n in internal], "d")
        .count()
        .where(F.col("count") > 1)
        .limit(1)
        .count()
    ):
        raise ValueError("Duplicate entity/date history records")

    def cutoff(k):
        return (
            F.add_months(F.col("r"), -k)
            if unit == "months"
            else F.date_sub(F.col("r"), k * (7 if unit == "weeks" else 1))
        )

    condition = reduce(
        lambda a, b: a & b, [F.col(n) == F.col("h" + n) for n in internal]
    )
    condition = condition & (F.col("d") <= F.col("r"))
    if max_staleness_days is not None:
        # Bound join expansion without discarding observations needed by windows.
        earliest = F.date_sub(cutoff(max([0, *lags])), max_staleness_days)
        if windows:
            earliest = F.least(earliest, cutoff(max(windows)))
        condition = condition & (F.col("d") >= earliest)
    joined = refs.join(hist, condition, "left")
    expressions = []
    for i, score in enumerate(scores):
        value = F.col(f"v{i}")
        for k in [0, *lags]:
            eligible = (F.col("d") <= cutoff(k)) & value.isNotNull()
            if max_staleness_days is not None:
                eligible = eligible & (
                    F.col("d") >= F.date_sub(cutoff(k), max_staleness_days)
                )
            latest = F.max(F.when(eligible, F.struct(F.col("d"), value.alias("value"))))
            name = f"{score}__" + ("current" if k == 0 else f"lag_{k}_{unit}")
            lookup_date = latest["d"]
            latency = (
                F.datediff(F.col("r"), lookup_date).alias(name + "_date_diff_days")
                if _date_diff
                else lookup_date.alias(name + "_date")
            )
            expressions.extend([latest["value"].alias(name), latency])
        for k in windows:
            selected = F.when(F.col("d") > cutoff(k), value)
            for agg in set(aggs) | {"mean"}:
                function = {
                    "mean": F.avg,
                    "median": F.median,
                    "std": F.stddev_samp,
                    "max": F.max,
                    "min": F.min,
                    "sum": F.sum,
                    "count": F.count,
                }[agg]
                expressions.append(
                    function(selected).alias(f"{score}__{agg}_{k}_{unit}")
                )
    features = joined.groupBy(*internal, "r").agg(*expressions)
    for score in scores:
        for k in windows:
            features = features.withColumn(
                f"{score}__current_minus_mean_{k}_{unit}",
                _col(f"{score}__current") - _col(f"{score}__mean_{k}_{unit}"),
            )
    left, right = spine.alias("spine"), features.alias("features")
    condition = reduce(
        lambda a, b: a & b,
        [left[k] == right[n] for k, n in zip(keys, internal, strict=True)],
    )
    condition = condition & (left[reference_col].cast("date") == right["r"])
    return left.join(right, condition, "left").select(
        *[left[c] for c in spine.columns], *[right[c] for c in output]
    )


def build_features(
    df_base: DataFrame,
    df_spine: DataFrame,
    *,
    key_cols: str | Sequence[str],
    date_col: str = "observation_date",
    reference_date_col: str = "dt_ref",
    value_cols: str | Sequence[str],
    period_unit: str = "months",
    windows: Sequence[int] = (3, 6, 9),
    lags: Sequence[int] = (1, 2, 8),
    aggregations: Sequence[str] = AGGREGATIONS,
    max_staleness_days: int | None = None,
    relative_historico: bool = False,
    ref_col: str | None = None,
) -> DataFrame:
    """Enrich df_spine with point-in-time features sourced only from df_base.

    df_base contains key_cols, date_col, and numeric value_cols. df_spine
    contains key_cols and reference_date_col; its existing value columns are
    preserved but never used to calculate features. Each spine row uses its own
    reference date, including multiple dates per entity and duplicate rows.

    Current and lag values are the latest non-null base values on/before their
    cutoffs. Windows are (reference minus k periods, reference], with calendar
    month arithmetic. Lookup date_diff_days features measure calendar days from
    the observation to the spine reference date; no lookup dates are returned.
    max_staleness_days is measured from each lookup cutoff.
    Sample std is null for fewer than two
    values; empty windows have null statistics and zero counts. Differences are
    current minus window mean. Duplicate base entity/date records are rejected.

    relative_historico adds comparisons with requested lags and level statistics.
    ref_col adds same-observation value/ref_col series to the temporal engine.
    Both options together also compare ratio series with their own histories.
    Zero/null denominators produce null; ratios stay on the ratio scale.

    Returns df_spine with its original rows and columns plus feature columns.
    See docs/temporal.md for naming, validation, and backend execution details.
    """
    keys, scores, all_scores, windows, lags, aggregations, ratios, specs = build_config(
        df_base,
        df_spine,
        key_cols,
        value_cols,
        date_col,
        reference_date_col,
        period_unit,
        windows,
        lags,
        aggregations,
        max_staleness_days,
        relative_historico,
        ref_col,
    )
    from pyspark.sql import functions as F
    from pyspark.sql.types import NumericType

    if ref_col is not None and not isinstance(
        df_base.schema[ref_col].dataType, NumericType
    ):
        raise ValueError("ref_col must be numeric")
    if any(not isinstance(df_base.schema[s].dataType, NumericType) for s in scores):
        raise ValueError("Value columns must be numeric")

    def divide(left, right):
        # Null the denominator before division, including under Spark ANSI mode.
        return left / F.when((right != 0) & ~F.isnan(right), right)

    if ratios:
        df_base = df_base.select(
            "*",
            *[
                divide(_col(source).cast("double"), _col(ref_col).cast("double")).alias(
                    name
                )
                for name, source in ratios.items()
            ],
        )
    result = temporal_features(
        spine=df_spine,
        history=df_base,
        entity_cols=keys,
        score_cols=all_scores,
        reference_col=reference_date_col,
        observation_col=date_col,
        unit=period_unit,
        windows=windows,
        lags=lags,
        aggregations=aggregations,
        max_staleness_days=max_staleness_days,
        _date_diff=True,
    )
    extra = []
    for name, operation, left, right in specs:
        a, b = _col(left), _col(right)
        value = (
            a - b
            if operation == "diff"
            else 100 * (a - b)
            if operation == "diff_pp"
            else 100 * divide(a - b, b)
            if operation == "pct_change"
            else divide(a, F.abs(b))
            if operation == "abs_ratio"
            else divide(a, b)
        )
        extra.append(value.alias(name))
    return result.select("*", *extra) if extra else result


def build_autoregressive_features(
    df_base: DataFrame,
    df_spine: DataFrame,
    *,
    key_cols: str | Sequence[str] = ("cnpj",),
    date_col: str = "bureau_date",
    reference_date_col: str = "dt_ref",
    value_cols: str | Sequence[str] = ("score",),
    period_unit: str = "months",
    windows: Sequence[int] = (3, 6, 9),
    lags: Sequence[int] = (1, 2, 8),
    aggregations: Sequence[str] = AGGREGATIONS,
    max_staleness_days: int | None = None,
) -> DataFrame:
    """Deprecated alias preserving the original column defaults and semantics."""
    warnings.warn(
        "build_autoregressive_features is deprecated; use build_features with "
        "explicit key_cols, value_cols, and date_col instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return temporal_features(
        history=df_base,
        spine=df_spine,
        entity_cols=key_cols,
        observation_col=date_col,
        reference_col=reference_date_col,
        score_cols=value_cols,
        unit=period_unit,
        windows=windows,
        lags=lags,
        aggregations=aggregations,
        max_staleness_days=max_staleness_days,
    )


__all__ = [
    "build_autoregressive_features",
    "build_features",
    "simulate_bureau_history",
    "temporal_features",
]
