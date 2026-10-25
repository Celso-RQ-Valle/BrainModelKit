# Numeric time-series features and bureau simulation

[Documentation home](README.md) ? [Backend journeys](../README.md)

Both `brainmodelkit.pandas` and `brainmodelkit.pyspark` export
`simulate_bureau_history` and `build_features`. Existing APIs,
including `temporal_features`, remain available unchanged.
Install the corresponding backend extra. The root package does not import either
backend dependency.

```python
import pandas as pd
from brainmodelkit.pandas import (
    build_features,
    simulate_bureau_history,
)

customers = pd.DataFrame({"cnpj": ["00000000000001", "00000000000002"]})
df_base = simulate_bureau_history(
    customers,
    entity_cols="cnpj",
    initial_date="2025-01-01",
    final_date="2026-10-13",
    frequency="monthly",
    seed=42,
    late_entry_fraction=0.1,
    missing_customer_fraction=0.1,
    observation_col="bureau_date",
    score_col="score",
)
df_spine = customers.merge(
    pd.DataFrame({"dt_ref": pd.to_datetime(["2026-09-13", "2026-10-13"])}),
    how="cross",
)
enriched = build_features(
    df_base,
    df_spine,
    key_cols=["cnpj"],
    date_col="bureau_date",
    reference_date_col="dt_ref",
    value_cols=["score"],
    period_unit="months",
    windows=[3, 6, 9],
    lags=[1, 2, 8],
    aggregations=["mean", "median", "max", "min", "std", "sum", "count"],
    max_staleness_days=None,
)
print(
    enriched[
        [
            "cnpj",
            "dt_ref",
            "score__current",
            "score__current_date_diff_days",
            "score__lag_1_months",
            "score__mean_3_months",
        ]
    ]
)
```

For Spark, import the same names from `brainmodelkit.pyspark` and pass Spark
DataFrames instead. Both APIs accept a single key name or a sequence of key names.
Composite identifiers and string identifiers with leading zeros are retained;
the simulator deduplicates customers by their keys. Keys and dates cannot be null.
Use matching key types in the spine and history. Dates are calendar dates: ISO
`YYYY-MM-DD` strings, date objects, or timezone-naive midnight timestamps.
Normalize timestamps/timezones before calling these APIs; Spark timestamps use
the session timezone. Pandas preserves the spine index and order; Spark has no
implicit row ordering. Both preserve spine columns, duplicate rows, and cardinality.

## Feature inputs

`build_features(df_base, df_spine, ...)` takes the historical source
**first** and the rows to enrich **second**:

- `df_base`: historical source, containing `key_cols`, observation dates in
  `date_col`, and numeric `value_cols`. All calculated feature values come from
  this table.
- `df_spine`: rows to enrich, containing `key_cols` and reference dates in
  `reference_date_col`. Any other columns, including existing score values, are
  preserved but are never used as source values for the features.

For each spine row, match all `key_cols` against the base and use that row's
reference date to determine the lookups and rolling windows. Multiple reference
dates for one customer and duplicate spine rows are supported. For composite
entities, pass e.g. `key_cols=["cnpj", "branch"]` on both backends.

`key_cols` and `value_cols` are required: select any numeric measures, such as
revenue, balances, credit limits, transaction amounts, or scores. Values are not
clipped or normalized. Defaults are `date_col="observation_date"`,
`reference_date_col="dt_ref"`, and `period_unit="months"`.
Lists and single strings are accepted for column names.

### Migration

`build_autoregressive_features` remains available on both backends and emits a
`DeprecationWarning` at the caller. It preserves its original defaults:
`key_cols=("cnpj",)`, `date_col="bureau_date"`, `value_cols=("score",)`.
Replace its name with `build_features` and pass these three arguments explicitly
to preserve a call that previously relied on defaults. All other defaults and
temporal semantics are unchanged. The legacy APIs retain lookup-date columns;
`build_features` replaces them with numeric `_date_diff_days` columns measured
from the observation date to the spine reference date (including for lags).
`simulate_bureau_history` remains a separate domain-specific data generator.

The original `temporal_features(spine, history, entity_cols, score_cols, ...)`
continues to work with its original argument order and defaults. The new API
delegates to that same backend implementation, mapping `df_base` to `history`,
`df_spine` to `spine`, `key_cols` to `entity_cols`, `value_cols` to `score_cols`,
`date_col` to `observation_col`, `reference_date_col` to `reference_col`, and
`period_unit` to `unit`. Temporal values and window semantics are identical;
lookup dates become latency features in `build_features`.

## Simulation

`frequency` is `daily`, `weekly`, or `monthly`. Daily and weekly dates start on
`initial_date`, advancing by one or seven days. Monthly dates are calendar month
ends. Only dates inside the inclusive initial/final range are emitted: January 1
through December 31, 2025 produces 12 observations for complete customers;
January 15 through February 15 produces January 31 only. Ranges without an aligned
date produce empty history.

Default output columns are `observation_date` and `bureau_score`; customize them
with `observation_col` and `score_col`. Bounds default to 0 and 1000. Each customer
has a beta-distributed baseline and an AR(1) path reverting toward that baseline:
`baseline + persistence * (previous - baseline) + normal_noise`, clipped to bounds.
`persistence=0.85` and `noise_std=25` are configurable, including zero noise.

Customers are assigned reproducibly to disjoint cohorts: `late_entry_fraction`
(default 0.1) enters at a uniformly selected period after the first;
`missing_customer_fraction` (default 0.1) can miss each period with
`missing_period_probability` (default 0.15). Remaining customers have full history.
The first two fractions must sum to less than 0.5, so a majority is complete in
expectation; realized proportions fluctuate, especially for small samples.
A single-period range cannot have late entrants. Missing-period customers can
have no records. The latent score path evolves even during missing periods.
Set both cohort fractions to zero to guarantee complete histories.

Seeded per-key generators make results independent of input ordering and Spark
partitioning, with backend parity for equivalent scalar key values/types. Preserve
identifier types rather than casting identifiers to numbers. The seed must be an
integer, fractions/persistence must be in [0, 1], bounds finite and increasing,
and noise finite and nonnegative.

## Temporal semantics

`period_unit` (`unit` in the original API) is `days`, `weeks` (seven days), or
`months` (calendar subtraction, clipping
to the last valid day). Window lengths and lag offsets must be unique positive
integers; empty lists are allowed. For **each numeric value separately**:

- Current: latest non-null score at or before the reference date.
- Lag k: latest non-null score at or before reference minus k periods.
- Window k: all non-null scores in `(reference minus k periods, reference]`.
- `current_minus_mean`: current score **minus** the window mean, even if mean
  is not among the requested aggregation outputs.

Every boundary is anchored to the spine reference date, never to the current
observation date. Future observations are excluded. For October 13, 2026, a latest
October 5 observation is current. Monthly lags 1, 2, and 8 use September 13,
August 13, and February 13 cutoffs. The eight-month window includes every
observation after February 13 through October 13, not just lag snapshots.
March 31 minus one month is February 28 (29 in a leap year).

Names use double underscores between the original score and feature:
`score__current`, `score__current_date_diff_days`, `score__lag_2_months`,
`score__lag_2_months_date_diff_days`, `score__mean_6_months`,
`score__current_minus_mean_6_months`. Existing spine column collisions fail.

Supported aggregations (also the defaults) are `mean`, exact `median`, `max`,
`min`, `std`, `sum`, and `count`. NaN/null scores are ignored. Empty windows have
null statistics, including sum, and count zero. Standard deviation is the sample
standard deviation (`ddof=1`), null with fewer than two observations. Unavailable
lookups and their latency indicators are null. `max_staleness_days`, when supplied,
is a nonnegative integer: each current/lag observation must be no more than that
many calendar days before **its own lookup cutoff**, inclusive. It does not filter
rolling windows. Latency indicators measure `reference_date - observation_date`
in calendar days,
including for lag lookups; they do not use the lag cutoff. Same-day observations
have latency zero. No generated lookup dates are returned by `build_features`;
original spine columns, including the reference date, remain unchanged.

Duplicate history entity/date records always raise `ValueError`, including exact
duplicates and duplicates with disjoint non-null scores. Aggregate or resolve
duplicates using an explicit business rule before calling. The APIs never choose
an arbitrary record. Duplicate spine entity/reference rows are fully preserved.

## Optional relationship families

`build_features` adds two keyword-only options on both backends:
`relative_historico: bool = False` and `ref_col: str | None = None`.
Existing calls and the legacy APIs retain their outputs. Neither option changes
window boundaries, as-of lookups, duplicate detection, or staleness rules.

| Options | Additional features |
| --- | --- |
| Both disabled | None; existing `current_minus_mean` remains for every window. |
| `relative_historico=True` | Historical comparisons for each `value_cols` series. |
| `ref_col="credit_limit"` | Temporal features for each `value_cols / credit_limit` series. |
| Both enabled | Both families, including historical comparisons of ratio series. |

### Own-history comparisons

Let `C` be the latest valid current value, `H` the requested lag or window
statistic, and `M`, `S`, `lo`, `hi` the window mean, sample std, minimum, maximum.
For every requested lag and every **enabled** level aggregation (`mean`, `median`,
`min`, `max`), the historical relationships are:

| Formula | Example name for `balance`, three-month mean or one-month lag |
| --- | --- |
| `C - H` | `balance__current_vs_lag_1_months_diff` |
| `C - M` | `balance__current_minus_mean_3_months` (existing name, no duplicate `_diff`) |
| `C / H` | `balance__current_vs_mean_3_months_ratio` |
| `100 * (C - H) / H` | `balance__current_vs_lag_1_months_pct_change` |
| `hi - lo`, if both `min`, `max` enabled | `balance__range_3_months` |
| `(C - M) / S`, if both `mean`, `std` enabled | `balance__standardized_deviation_3_months` |
| `S / abs(M)`, if both `mean`, `std` enabled | `balance__coefficient_of_variation_3_months` |

For median/min/max differences, use e.g.
`balance__current_vs_median_3_months_diff`. Names use the requested period unit,
including `days` and `weeks`. Only the requested lags/windows are considered;
there are no automatic comparisons between pairs of windows.

`aggregations=["mean"]` adds no median, range, standardized-deviation, or
coefficient-of-variation calculations. `sum`, `count`, and `std` remain ordinary
aggregations and are not treated as historical levels for difference/ratio/change
comparisons. The existing `current_minus_mean` is always retained, even when
`mean` is not selected; its internal mean calculation is the backward-compatible
exception to the selected-statistics rule. It does not enable other mean-based
relationships unless `mean` is requested.

### Reference-column series

`ref_col` must name a numeric column in `df_base`, distinct from the key/date
columns. For every requested source other than the reference column, form a
same-row historical ratio, named `<value>__over__<ref_col>`. For example:

```text
used_balance__over__credit_limit__current
used_balance__over__credit_limit__current_date_diff_days
used_balance__over__credit_limit__lag_1_months
used_balance__over__credit_limit__mean_3_months
used_balance__over__credit_limit__count_3_months
used_balance__over__credit_limit__current_minus_mean_3_months
```

This derived series passes through the same temporal engine, with the same
requested windows, lags, aggregations, and observation-age indicators as the
original series. Historical numerator and denominator always come from the same
entity and observation date. No join pairs values from different dates and no
historical numerator is divided by a current denominator. If `ref_col` also
appears in `value_cols`, its individual features remain, but no self-ratio is made.

January balance 20 / limit 100 gives **0.20**; February balance 60 / limit 200
gives **0.30**. The two-observation mean ratio is **0.25**, not
`mean(balance) / mean(limit) = 40 / 150`. Ratios remain on the ratio scale: 0.20
represents 20%. Every requested aggregation is supported; the sum is **0.50**
here, a sum of ratio observations, **not accumulated utilization**.

When both options are enabled, ratio series also receive the own-history
comparisons and one additional scale-specific difference per requested lag and
enabled level statistic:

| Interpretation (current 0.30, lag 0.20) | Value | Suffix after `used_balance__over__credit_limit__` |
| --- | ---: | --- |
| Difference in ratio, `C - H` | 0.10 | `current_vs_lag_1_months_diff` |
| Difference in percentage points, `100 * (C - H)` | 10 p.p. | `current_vs_lag_1_months_diff_pp` |
| Ratio, `C / H` | 1.50 | `current_vs_lag_1_months_ratio` |
| Percentage change, `100 * (C - H) / H` | 50% | `current_vs_lag_1_months_pct_change` |

For window means, the absolute ratio difference keeps the existing
`current_minus_mean_<k>_<unit>` name. Its percentage-point counterpart is
`current_vs_mean_<k>_<unit>_diff_pp`. Aggregations are not duplicated on a
percentage scale. `ref_col` alone does not enable these optional comparisons.

### Missing values and validation

A zero, NaN, or null denominator produces a null ratio; a missing numerator also
produces null. Spark division is guarded before evaluation, including with ANSI
mode enabled. The library never fills nulls with zero. Derived-series current/lag
values use the latest **valid ratio**, not the latest numerator or denominator
individually, and their observation age refers to that actual historical row.
A more recent invalid ratio does not hide an earlier valid ratio; staleness still
applies from each lookup cutoff. Future observations are excluded.

Windows count only valid ratios. Empty-window statistics, including sum, are
null with count zero. Sample std is null with fewer than two valid observations.
Standardized deviation is null for zero std, and coefficient of variation is
null for zero mean. Comparisons with a missing current, lag, or statistic are null.

Missing/non-numeric reference columns fail with a `ValueError`. Duplicate
entity/date records remain errors even if one row has an invalid ratio. Generated
series names cannot collide with existing base or spine columns; generated output
names cannot collide with spine columns or one another. Original spine columns,
rows, and duplicates are retained; pandas also preserves its index/order.

For an executable example on the project's simulated credit export, run
`python examples/temporal/feature_relationships.py simulated_credit.csv` from a
checkout. The [README](../README.md#spark-prepare-data) generates the CSV.

## Execution and scale

Pandas sorts history once, groups non-null values by entity and source column,
uses binary searches for lookup/window boundaries, and caches each distinct
entity/reference result. Window statistics use native pandas reductions;
it is intended for in-memory datasets. Spark validates with distributed actions,
deduplicates reference queries, performs an entity equality join with an as-of
date predicate, aggregates, and joins features back to the original spine. With
a staleness limit, the join also excludes history older than every requested
lookup and window. It
never converts to pandas or collects history/customer records to the driver.
Spark simulation uses a Python UDF on workers, generating one period array per
customer; memory per row grows with the requested number of periods.

The feature join can still produce many pairs for entities with long histories
and many reference dates. All earlier history is eligible for unlimited-staleness
lookups; exact median also has group-size memory costs. Partition by entity and
process bounded spine batches for large workloads. Validation adds scans/shuffles.
Spark output order is unspecified; use `orderBy` explicitly when needed.

Reference ratios are projected once before the temporal engine. Spark uses native
expressions and includes all original/derived series in the existing history join
and grouped aggregation; it does not collect historical records or call
`toPandas()`. Historical relationships are a final projection over computed
features, with no additional history joins or window aggregations. pandas uses
native Series arithmetic and shares the same feature configuration/naming rules.
Only enabled relationship families are constructed. The unchanged legacy mean
needed for `current_minus_mean` remains the sole implicit window statistic.
