"""Pandas bureau histories and point-in-time features. See docs/temporal.md."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from datetime import date

import pandas as pd
from pandas.api.types import is_numeric_dtype

from brainmodelkit._temporal import (
    AGGREGATIONS,
    as_date,
    columns,
    feature_config,
    shift,
    simulate_entity,
    simulation_config,
)


def simulate_bureau_history(
    customers: pd.DataFrame,
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
) -> pd.DataFrame:
    """Simulate bounded persistent scores; monthly dates align to month ends.

    Daily/weekly schedules start on initial_date. Only dates within the inclusive
    range are emitted. Duplicate customer keys are deduplicated; null keys fail.
    Incomplete cohorts are disjoint; their combined probability must be < 0.5.
    """
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
    if customers[keys].isna().any().any():
        raise ValueError("Customer keys must not be null")
    records = []
    for key in customers[keys].drop_duplicates().itertuples(index=False, name=None):
        for day, score in simulate_entity(
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
        ):
            records.append((*key, day, score))
    result = pd.DataFrame(records, columns=[*keys, observation_col, score_col])
    for key in keys:
        result[key] = result[key].astype(customers[key].dtype)
    result[observation_col] = pd.to_datetime(result[observation_col])
    result[score_col] = result[score_col].astype(float)
    return result


def temporal_features(
    spine: pd.DataFrame,
    history: pd.DataFrame,
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
) -> pd.DataFrame:
    """Enrich the spine using only history on/before each reference date.

    Windows are (reference - k periods, reference]; lags are as-of lookups.
    Each score uses its latest non-null value. Staleness is measured from the
    lookup cutoff. Sample std (ddof=1) is null for fewer than two values.
    Duplicate entity/date history records raise ValueError, even if identical.
    """
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
    )
    columns(spine, [*keys, reference_col])
    columns(history, [*keys, observation_col, *scores])
    if set(output) & set(spine.columns):
        raise ValueError("Feature columns already exist in spine")
    for frame, day in ((spine, reference_col), (history, observation_col)):
        if frame[[*keys, day]].isna().any().any():
            raise ValueError("Keys and dates must not be null")
    if any(not is_numeric_dtype(history[s]) for s in scores):
        raise ValueError("Value columns must be numeric")
    hist = history[[*keys, observation_col, *scores]].copy()
    hist[observation_col] = hist[observation_col].map(as_date)
    if hist.duplicated([*keys, observation_col]).any():
        raise ValueError("Duplicate entity/date history records")
    hist = hist.sort_values(observation_col)
    groups = {
        key if isinstance(key, tuple) else (key,): {
            value: group.loc[group[value].notna(), [observation_col, value]]
            for value in scores
        }
        for key, group in hist.groupby(keys, sort=False, observed=True)
    }
    cache, records = {}, []
    for row in spine[[*keys, reference_col]].itertuples(index=False, name=None):
        key, ref = tuple(row[:-1]), as_date(row[-1])
        token = (*key, ref)
        if token not in cache:
            group = groups.get(key)
            features = {}
            for score in scores:
                valid = group[score] if group is not None else hist.iloc[:0]
                dates = valid[observation_col]
                for k in [0, *lags]:
                    cutoff = shift(ref, k, unit)
                    position = dates.searchsorted(cutoff, side="right") - 1
                    available = position >= 0
                    if available and max_staleness_days is not None:
                        available = dates.iloc[position] >= shift(
                            cutoff, max_staleness_days, "days"
                        )
                    name = f"{score}__" + ("current" if k == 0 else f"lag_{k}_{unit}")
                    features[name] = valid[score].iloc[position] if available else None
                    features[name + "_date"] = (
                        dates.iloc[position] if available else None
                    )
                for k in windows:
                    lower = dates.searchsorted(shift(ref, k, unit), side="right")
                    upper = dates.searchsorted(ref, side="right")
                    values = valid[score].iloc[lower:upper]
                    for agg in aggs:
                        value = (
                            len(values)
                            if agg == "count"
                            else getattr(values, agg)()
                            if len(values)
                            else None
                        )
                        features[f"{score}__{agg}_{k}_{unit}"] = value
                    current = features[f"{score}__current"]
                    features[f"{score}__current_minus_mean_{k}_{unit}"] = (
                        current - values.mean()
                        if current is not None and len(values)
                        else None
                    )
            cache[token] = features
        records.append(cache[token])
    result = spine.copy()
    features = pd.DataFrame(records, columns=output)
    for name in output:
        values = features[name]
        result[name] = (
            pd.to_datetime(values).to_numpy()
            if name.endswith("_date")
            else pd.to_numeric(values).to_numpy()
        )
    return result


def build_features(
    df_base: pd.DataFrame,
    df_spine: pd.DataFrame,
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
) -> pd.DataFrame:
    """Enrich df_spine with point-in-time features sourced only from df_base.

    df_base contains key_cols, date_col, and numeric value_cols. df_spine
    contains key_cols and reference_date_col; its existing value columns are
    preserved but never used to calculate features. Each spine row uses its own
    reference date, including multiple dates per entity and duplicate rows.

    Current and lag values are the latest non-null base values on/before their
    cutoffs. Windows are (reference minus k periods, reference], with calendar
    month arithmetic. Observation dates accompany lookups; max_staleness_days
    is measured from each lookup cutoff. Sample std is null for fewer than two
    values; empty windows have null statistics and zero counts. Differences are
    current minus window mean. Duplicate base entity/date records are rejected.

    Returns df_spine with its original rows and columns plus feature columns.
    See docs/temporal.md for naming, validation, and backend execution details.
    """
    return temporal_features(
        spine=df_spine,
        history=df_base,
        entity_cols=key_cols,
        score_cols=value_cols,
        reference_col=reference_date_col,
        observation_col=date_col,
        unit=period_unit,
        windows=windows,
        lags=lags,
        aggregations=aggregations,
        max_staleness_days=max_staleness_days,
    )


def build_autoregressive_features(
    df_base: pd.DataFrame,
    df_spine: pd.DataFrame,
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
) -> pd.DataFrame:
    """Deprecated alias preserving the original column defaults and semantics."""
    warnings.warn(
        "build_autoregressive_features is deprecated; use build_features with "
        "explicit key_cols, value_cols, and date_col instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_features(
        df_base,
        df_spine,
        key_cols=key_cols,
        date_col=date_col,
        reference_date_col=reference_date_col,
        value_cols=value_cols,
        period_unit=period_unit,
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
