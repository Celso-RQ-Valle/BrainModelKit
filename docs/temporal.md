# Bureau history and point-in-time features

Both `brainmodelkit.pandas` and `brainmodelkit.pyspark` export
`simulate_bureau_history` and `build_autoregressive_features`. Existing APIs,
including `temporal_features`, remain available unchanged.
Install the corresponding backend extra. The root package does not import either
backend dependency.

```python
import pandas as pd
from brainmodelkit.pandas import (
    build_autoregressive_features,
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
enriched = build_autoregressive_features(
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
            "score__current_date",
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

`build_autoregressive_features(df_base, df_spine, ...)` takes the historical source
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

Defaults are `key_cols=("cnpj",)`, `date_col="bureau_date"`,
`reference_date_col="dt_ref"`, `value_cols=("score",)`, and
`period_unit="months"`. Lists and single strings are accepted for column names.
The simulator's original column defaults remain unchanged; the example explicitly
names its outputs to match this API.

The original `temporal_features(spine, history, entity_cols, score_cols, ...)`
continues to work with its original argument order and defaults. The new API
delegates to that same backend implementation, mapping `df_base` to `history`,
`df_spine` to `spine`, `key_cols` to `entity_cols`, `value_cols` to `score_cols`,
`date_col` to `observation_col`, `reference_date_col` to `reference_col`, and
`period_unit` to `unit`. Temporal semantics and feature names are identical.

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
integers; empty lists are allowed. For **each numeric score separately**:

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
`score__current`, `score__current_date`, `score__lag_2_months`,
`score__lag_2_months_date`, `score__mean_6_months`,
`score__current_minus_mean_6_months`. Existing spine column collisions fail.

Supported aggregations (also the defaults) are `mean`, exact `median`, `max`,
`min`, `std`, `sum`, and `count`. NaN/null scores are ignored. Empty windows have
null statistics, including sum, and count zero. Standard deviation is the sample
standard deviation (`ddof=1`), null with fewer than two observations. Unavailable
lookups and their observation dates are null. `max_staleness_days`, when supplied,
is a nonnegative integer: each current/lag observation must be no more than that
many calendar days before **its own lookup cutoff**, inclusive. It does not filter
rolling windows. Dates expose staleness even when no limit is applied.

Duplicate history entity/date records always raise `ValueError`, including exact
duplicates and duplicates with disjoint non-null scores. Aggregate or resolve
duplicates using an explicit business rule before calling. The APIs never choose
an arbitrary record. Duplicate spine entity/reference rows are fully preserved.

## Execution and scale

Pandas groups history by entity and caches each distinct entity/reference result;
it is intended for in-memory datasets. Spark validates with distributed actions,
deduplicates reference queries, performs an entity equality join with an as-of
date predicate, aggregates, and joins features back to the original spine. It
never converts to pandas or collects history/customer records to the driver.
Spark simulation uses a Python UDF on workers, generating one period array per
customer; memory per row grows with the requested number of periods.

The feature join can still produce many pairs for entities with long histories
and many reference dates. All earlier history is eligible for unlimited-staleness
lookups; exact median also has group-size memory costs. Partition by entity and
process bounded spine batches for large workloads. Validation adds scans/shuffles.
Spark output order is unspecified; use `orderBy` explicitly when needed.
