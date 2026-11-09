"""Backend-independent bureau simulation and temporal parameter contracts."""

from __future__ import annotations

import calendar
import hashlib
import math
import random
import re
from datetime import date, datetime, timedelta
from numbers import Real

AGGREGATIONS = ("mean", "median", "max", "min", "std", "sum", "count")


def names(values, label):
    try:
        values = [values] if isinstance(values, str) else list(values)
    except TypeError as error:
        raise ValueError(
            f"{label} must be a non-empty sequence of column names"
        ) from error
    if not values or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{label} must contain non-empty column names")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
    return values


def feature_name(score, suffix, modern=True):
    """Normalize generated separators only; never rename input columns."""
    if not modern:
        return f"{score}__{suffix}"
    return re.sub(r"_+", "_", f"{score}_{suffix}")


def custom_config(base, spine, value_cols, create_cols, nullable, is_numeric):
    """Validate the expression tree against the original schema, without data scans."""
    columns(base, [])
    columns(spine, [])
    if type(nullable) is not bool:
        raise ValueError("nullable must be a bool")
    if create_cols is None:
        create_cols = {}
    if not isinstance(create_cols, dict):
        raise ValueError("create_cols must be a dictionary or None")
    scores = names(value_cols, "value_cols")
    operands = set()
    for name in create_cols:
        if not isinstance(name, str) or not name:
            raise ValueError("Custom column names must be non-empty strings")
        if name in base.columns or name in spine.columns:
            raise ValueError(
                f"Custom column {name!r} conflicts with an existing column"
            )
        if name not in scores:
            raise ValueError(f"Custom column {name!r} must appear in value_cols")

    def validate(expr):
        if not isinstance(expr, dict):
            raise ValueError("Each expression must be a dictionary")
        if set(expr) == {"col"}:
            name = expr["col"]
            if not isinstance(name, str):
                raise ValueError("Column references must be strings")
            if name in create_cols:
                raise ValueError(
                    f"Cannot reference newly declared custom column {name!r}"
                )
            if name not in base.columns:
                raise ValueError(f"Expression column {name!r} is missing from df_base")
            if not is_numeric(name):
                raise ValueError(f"Expression column {name!r} must be numeric")
            operands.add(name)
        elif set(expr) == {"lit"}:
            value = expr["lit"]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError("Expression literals must be finite numeric values")
        elif set(expr) == {"op", "args"}:
            op, args = expr["op"], expr["args"]
            if not isinstance(op, str) or op not in (
                "sum",
                "subtract",
                "multiply",
                "divide",
            ):
                raise ValueError(f"Unknown expression operation: {op!r}")
            if (
                not isinstance(args, list)
                or len(args) < 2
                or (op in ("subtract", "divide") and len(args) != 2)
            ):
                raise ValueError(f"Invalid operand count for {op!r}")
            for arg in args:
                validate(arg)
        else:
            raise ValueError("Malformed expression: expected col, lit, or op and args")

    for expr in create_cols.values():
        validate(expr)
    columns(base, [s for s in scores if s not in create_cols])
    return scores, create_cols, operands


def custom_expression(expr, column, literal, divide):
    """Compile a validated tree with backend-native arithmetic operators."""
    if "col" in expr:
        return column(expr["col"])
    if "lit" in expr:
        return literal(expr["lit"])
    args = [custom_expression(arg, column, literal, divide) for arg in expr["args"]]
    value = args[0]
    for right in args[1:]:
        op = expr["op"]
        value = (
            value + right
            if op == "sum"
            else value - right
            if op == "subtract"
            else value * right
            if op == "multiply"
            else divide(value, right)
        )
    return value


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
            output.extend(
                [
                    feature_name(score, f"{suffix}", date_diff),
                    feature_name(score, f"{suffix}{date_suffix}", date_diff),
                ]
            )
        for k in windows:
            output.extend(
                feature_name(score, f"{a}_{k}_{unit}", date_diff) for a in aggregations
            )
            output.append(
                feature_name(score, f"current_minus_mean_{k}_{unit}", date_diff)
            )
    if len(set(output)) != len(output):
        raise ValueError("Generated feature names collide")
    return keys, scores, windows, lags, aggregations, output


def relationship_config(scores, windows, lags, aggs, unit):
    """Return backend-neutral (name, operation, left, right) expressions."""
    specs = []
    for score in scores:
        current = feature_name(score, "current")
        comparisons = [(f"lag_{k}_{unit}", False) for k in lags]
        comparisons += [
            (f"{agg}_{k}_{unit}", agg == "mean")
            for k in windows
            for agg in aggs
            if agg in ("mean", "median", "min", "max")
        ]
        for suffix, existing_mean in comparisons:
            other = feature_name(score, f"{suffix}")
            prefix = feature_name(score, f"current_vs_{suffix}")
            # The legacy current_minus_mean already supplies this difference.
            if not existing_mean:
                specs.append((prefix + "_diff", "diff", current, other))
            specs.extend(
                [
                    (prefix + "_ratio", "ratio", current, other),
                    (prefix + "_pct_change", "pct_change", current, other),
                ]
            )
        for k in windows:
            suffix = f"{k}_{unit}"
            mean = feature_name(score, f"mean_{suffix}")
            std = feature_name(score, f"std_{suffix}")
            if {"min", "max"} <= set(aggs):
                specs.append(
                    (
                        feature_name(score, f"range_{suffix}"),
                        "diff",
                        feature_name(score, f"max_{suffix}"),
                        feature_name(score, f"min_{suffix}"),
                    )
                )
            if {"mean", "std"} <= set(aggs):
                specs.extend(
                    [
                        (
                            feature_name(score, f"standardized_deviation_{suffix}"),
                            "ratio",
                            feature_name(score, f"current_minus_mean_{suffix}"),
                            std,
                        ),
                        (
                            feature_name(score, f"coefficient_of_variation_{suffix}"),
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
):
    """Validate optional families and reserve all names before backend work."""
    if type(relative_historico) is not bool:
        raise ValueError("relative_historico must be a bool")
    keys, scores, windows, lags, aggs, output = feature_config(
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
    specs = (
        relationship_config(scores, windows, lags, aggs, period_unit)
        if relative_historico
        else []
    )
    output += [spec[0] for spec in specs]
    if len(set(output)) != len(output):
        raise ValueError("Generated feature names collide")
    if conflicts := set(output) & (set(spine.columns) | set(base.columns)):
        raise ValueError(
            f"Feature columns already exist in base or spine: {sorted(conflicts)}"
        )
    return keys, scores, windows, lags, aggs, specs


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
