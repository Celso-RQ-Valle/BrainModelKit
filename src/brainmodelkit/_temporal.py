"""Backend-independent bureau simulation and temporal parameter contracts."""

from __future__ import annotations

import calendar
import hashlib
import math
import random
from datetime import date, datetime, timedelta

AGGREGATIONS = ("mean", "median", "max", "min", "std", "sum", "count")


def names(values, label):
    values = [values] if isinstance(values, str) else list(values)
    if not values or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{label} must contain non-empty column names")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
    return values


def columns(frame, required):
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("Duplicate column names are not supported")
    if missing := set(required) - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")


def as_date(value):
    if isinstance(value, datetime):
        if value.tzinfo is not None or value.time() != datetime.min.time():
            raise ValueError("Dates must be timezone-naive dates at midnight")
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Dates must be ISO dates or date objects") from error


def shift(value, n, unit):
    if unit != "months":
        return value - timedelta(days=n * (7 if unit == "weeks" else 1))
    month = value.year * 12 + value.month - 1 - n
    year, month = divmod(month, 12)
    month += 1
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


def feature_config(
    entity_cols,
    score_cols,
    reference_col,
    observation_col,
    unit,
    windows,
    lags,
    aggregations,
    max_staleness_days,
    *,
    date_diff=False,
):
    keys, scores = names(entity_cols, "entity_cols"), names(score_cols, "score_cols")
    if set(keys) & set(scores) or reference_col in keys or observation_col in keys:
        raise ValueError("Keys, dates, and scores must use distinct names")
    if observation_col in scores or reference_col in scores:
        raise ValueError("Scores cannot be date columns")
    if unit not in ("days", "weeks", "months"):
        raise ValueError("unit must be days, weeks, or months")
    windows, lags, aggregations = list(windows), list(lags), list(aggregations)
    for label, values in (("windows", windows), ("lags", lags)):
        if any(type(v) is not int or v <= 0 for v in values):
            raise ValueError(f"{label} must contain positive integers")
        if len(set(values)) != len(values):
            raise ValueError(f"{label} must be unique")
    if len(set(aggregations)) != len(aggregations) or set(aggregations) - set(
        AGGREGATIONS
    ):
        raise ValueError(f"aggregations must be unique members of {AGGREGATIONS}")
    if max_staleness_days is not None and (
        type(max_staleness_days) is not int or max_staleness_days < 0
    ):
        raise ValueError("max_staleness_days must be a non-negative integer")
    output = []
    for score in scores:
        for suffix in ["current", *(f"lag_{k}_{unit}" for k in lags)]:
            date_suffix = "_date_diff_days" if date_diff else "_date"
            output.extend([f"{score}__{suffix}", f"{score}__{suffix}{date_suffix}"])
        for k in windows:
            output.extend(f"{score}__{a}_{k}_{unit}" for a in aggregations)
            output.append(f"{score}__current_minus_mean_{k}_{unit}")
    if len(set(output)) != len(output):
        raise ValueError("Generated feature names collide")
    return keys, scores, windows, lags, aggregations, output


def relationship_config(scores, ratio_scores, windows, lags, aggs, unit):
    """Return backend-neutral (name, operation, left, right) expressions."""
    specs = []
    for score in scores:
        current = f"{score}__current"
        comparisons = [(f"lag_{k}_{unit}", False) for k in lags]
        comparisons += [
            (f"{agg}_{k}_{unit}", agg == "mean")
            for k in windows
            for agg in aggs
            if agg in ("mean", "median", "min", "max")
        ]
        for suffix, existing_mean in comparisons:
            other = f"{score}__{suffix}"
            prefix = f"{score}__current_vs_{suffix}"
            # The legacy current_minus_mean already supplies this difference.
            if not existing_mean:
                specs.append((prefix + "_diff", "diff", current, other))
            if score in ratio_scores:
                specs.append((prefix + "_diff_pp", "diff_pp", current, other))
            specs.extend(
                [
                    (prefix + "_ratio", "ratio", current, other),
                    (prefix + "_pct_change", "pct_change", current, other),
                ]
            )
        for k in windows:
            suffix = f"{k}_{unit}"
            mean = f"{score}__mean_{suffix}"
            std = f"{score}__std_{suffix}"
            if {"min", "max"} <= set(aggs):
                specs.append(
                    (
                        f"{score}__range_{suffix}",
                        "diff",
                        f"{score}__max_{suffix}",
                        f"{score}__min_{suffix}",
                    )
                )
            if {"mean", "std"} <= set(aggs):
                specs.extend(
                    [
                        (
                            f"{score}__standardized_deviation_{suffix}",
                            "ratio",
                            f"{score}__current_minus_mean_{suffix}",
                            std,
                        ),
                        (
                            f"{score}__coefficient_of_variation_{suffix}",
                            "abs_ratio",
                            std,
                            mean,
                        ),
                    ]
                )
    return specs


def build_config(
    base,
    spine,
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
):
    """Validate optional families and reserve all names before backend work."""
    if type(relative_historico) is not bool:
        raise ValueError("relative_historico must be a bool")
    keys, scores, windows, lags, aggs, _ = feature_config(
        key_cols,
        value_cols,
        reference_date_col,
        date_col,
        period_unit,
        windows,
        lags,
        aggregations,
        max_staleness_days,
        date_diff=True,
    )
    columns(base, [*keys, date_col, *scores])
    columns(spine, [*keys, reference_date_col])
    ratios = {}
    if ref_col is not None:
        if not isinstance(ref_col, str) or not ref_col:
            raise ValueError("ref_col must be a non-empty column name or None")
        if ref_col not in base.columns:
            raise ValueError(f"ref_col column {ref_col!r} is missing from df_base")
        if ref_col in [*keys, date_col, reference_date_col]:
            raise ValueError(
                "ref_col must be a numeric value column, not a key or date"
            )
        ratios = {
            f"{score}__over__{ref_col}": score for score in scores if score != ref_col
        }
        if set(ratios) & (set(base.columns) | set(spine.columns)):
            raise ValueError("Derived ratio series names collide with existing columns")
    all_scores = [*scores, *ratios]
    _, _, _, _, _, output = feature_config(
        keys,
        all_scores,
        reference_date_col,
        date_col,
        period_unit,
        windows,
        lags,
        aggs,
        max_staleness_days,
        date_diff=True,
    )
    specs = (
        relationship_config(all_scores, ratios, windows, lags, aggs, period_unit)
        if relative_historico
        else []
    )
    output += [spec[0] for spec in specs]
    if len(set(output)) != len(output):
        raise ValueError("Generated feature names collide")
    if set(output) & set(spine.columns):
        raise ValueError("Feature columns already exist in spine")
    # A derived input must not masquerade as another generated feature.
    if set(ratios) & set(output):
        raise ValueError("Derived ratio series names collide with generated features")
    return keys, scores, all_scores, windows, lags, aggs, ratios, specs


def simulation_config(
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
):
    keys = names(entity_cols, "entity_cols")
    if observation_col == score_col or {observation_col, score_col} & set(keys):
        raise ValueError("Output column names must be distinct")
    start, end = as_date(initial_date), as_date(final_date)
    if start > end:
        raise ValueError("initial_date must be on or before final_date")
    if frequency not in ("daily", "weekly", "monthly"):
        raise ValueError("frequency must be daily, weekly, or monthly")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    for label, value in (
        ("late_entry_fraction", late_entry_fraction),
        ("missing_customer_fraction", missing_customer_fraction),
        ("missing_period_probability", missing_period_probability),
        ("persistence", persistence),
    ):
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{label} must be between 0 and 1")
    if late_entry_fraction + missing_customer_fraction >= 0.5:
        raise ValueError("Incomplete customer fractions must sum to less than 0.5")
    if not all(math.isfinite(v) for v in (score_min, score_max, noise_std)):
        raise ValueError("Score bounds and noise_std must be finite")
    if score_min >= score_max or noise_std < 0:
        raise ValueError("Require score_min < score_max and noise_std >= 0")
    dates = []
    current = start
    if frequency == "monthly":
        current = start.replace(day=calendar.monthrange(start.year, start.month)[1])
    while current <= end:
        dates.append(current)
        if frequency == "monthly":
            next_month = current.replace(day=1) + timedelta(days=32)
            current = next_month.replace(
                day=calendar.monthrange(next_month.year, next_month.month)[1]
            )
        else:
            current += timedelta(days=7 if frequency == "weekly" else 1)
    return keys, dates


def simulate_entity(
    key, dates, seed, late, missing, probability, low, high, rho, noise
):
    """Stable per-key AR(1) path, independent of row order and Spark partitions."""
    digest = hashlib.sha256(repr((seed, tuple(key))).encode()).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    cohort = rng.random()
    entry = rng.randrange(1, len(dates)) if cohort < late and len(dates) > 1 else 0
    baseline = low + (high - low) * rng.betavariate(5, 3)
    score = baseline
    result = []
    for i, day in enumerate(dates):
        score = min(
            high, max(low, baseline + rho * (score - baseline) + rng.gauss(0, noise))
        )
        omitted = late <= cohort < late + missing and rng.random() < probability
        if i >= entry and not omitted:
            result.append((day, float(score)))
    return result
