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
