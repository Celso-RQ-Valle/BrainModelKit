# External-provider dataframe validation

Use `from brainmodelkit.pyspark import validate_external_dataframe` to compare a
provider's return with the original submission and inspect score stability.
Install the `pyspark` extra; install `matplotlib` for figures and `IPython` for
notebook display. Without plotting dependencies the metrics still work and the
report includes a warning. In notebooks the sections display automatically;
in scripts, use the returned dictionary and figures.

## Parameters

| Parameter | Meaning |
| --- | --- |
| `df_sent` | Original submitted Spark DataFrame. |
| `df_received` | Provider-returned Spark DataFrame; never modified or deduplicated. |
| `key_cols` | Nonempty list of unique column names; composite keys are supported. Corresponding key types must match. |
| `date_col` | Received date, timestamp, or strict `YYYY-MM-DD` string column. Timestamp months follow the Spark session timezone. |
| `score_col` | Received numeric score column. Nonnumeric types are rejected rather than coerced. |
| `baseline` | Optional `YYYYMM` string. `None` chooses the earliest valid received month. |

Malformed baseline arguments raise `ValueError`. A nonexistent cohort or a
baseline without finite scores makes stability unavailable, while the other
sections remain usable. The default never skips an early month with no scores.
Missing required columns, duplicate column names and incompatible types raise
`ValueError` before analysis.

## Return structure and report order

The return value is a dictionary. Tables are lists of dictionaries, figures are
Matplotlib `Figure` objects, counts are integers, and percentages use 0–100.
Zero-denominator percentages are `None`, not zero.

1. `matching`: `submitted` and `received` summaries, matched distinct keys,
   missing submitted keys, unexpected received keys, and match percentages.
2. `scores`: missing, invalid, valid, below-range and above-range counts plus
   `missing_percentage` over **all received rows**.
3. `monthly_volume`: chronological `cohort`, `count` table; `dates` separately
   reports `missing_dates` and `invalid_dates`.
4. `baseline`, `cutoffs`, `band_definitions`: selected cohort, nine frozen
   percentile cutoffs, and band limits/inclusivity.
5. `monthly_bands`: chronological `cohort`, `band`, `count`, `percentage` table.
   `stability_status` is `"available"` or an explanation of unavailability.
6. `histogram`: ten rows with `lower`, `upper`, `upper_inclusive`, `count`,
   `percentage`, and `density`.
7. `warnings`: data-quality and optional plotting dependency messages.

`figures` contains `monthly_volume`, `monthly_bands`, and `histogram` when
Matplotlib is installed. For example:

```python
report["figures"]["histogram"].savefig("provider-density.png", dpi=150)
```

## Matching and data quality

Coverage uses an inner join of distinct **valid** keys. Any SQL-null key
component invalidates the key. Submitted match percentage divides matched keys
by distinct valid submitted keys; received match percentage uses distinct valid
received keys. Missing/unexpected counts also concern valid keys only.

Each dataset summary includes `rows`, `distinct_keys` (including null-containing
combinations), `valid_distinct_keys`, `null_key_rows`, `duplicate_key_groups`,
and `duplicate_key_rows`. Duplicate groups have more than one row; duplicate
rows count excess rows beyond the first per group, including null-key groups.
These categories may overlap. No row is removed from score analyses: unmatched
records, duplicate records and null-key rows all contribute.

SQL null and numeric NaN are missing scores. Positive/negative infinity are
invalid, counted separately. Finite scores outside [0, 1000] are valid for
baseline/stability calculations but excluded from the histogram and reported
separately. Invalid scores in any valid-dated cohort make the eleven-band
stability analysis unavailable, preventing misleading percentages. Invalid
scores on undated rows do not prevent analysis of valid cohorts.

SQL-null dates and invalid dates are counted separately. Unsupported strings,
impossible dates and empty strings are invalid. These rows remain in overall
score/matching metrics and the histogram but cannot enter monthly analyses.

## Chart interpretation

The volume columns count every received row with a valid monthly date. Changes
may reflect provider coverage as well as changes in the submitted population.

The stability chart is a 100% stacked column per month, ordered `Missing`,
`D1`, ..., `D10`, using fixed colors. Its denominator includes every received
row in that month, including missing scores. Missing scores have their own band.
The baseline's finite, nonmissing scores determine p10 through p90 using Spark
`percentile_approx` with accuracy 10000 (relative rank error 1/10000).
Thresholds are calculated once and reused for every cohort:

- D1: `(-infinity, p10]`.
- D2 through D9: `(previous cutoff, current cutoff]`.
- D10: `(p90, infinity)` for finite scores.

Cutoff equality belongs to the first eligible band. Repeated cutoffs create
empty bands; identical scores are never split. All ten labels remain in the
table, even for tiny or constant baselines. Ties can prevent baseline bands from
containing exactly 10% each. A constant baseline puts all baseline scores in D1.
A shift toward higher bands indicates higher scores relative to the same
baseline thresholds; this alone does not establish improved model quality.

The density histogram uses `[0,100)`, `[100,200)`, ..., `[900,1000]`.
Exactly 100 belongs to the second bin; exactly 1000 belongs to the last bin.
Density is `count / (in-range count * 100)`, so total bar area is one. Bin
percentages use only finite in-range scores. Missing, invalid and out-of-range
scores are never clipped into bins. An empty state explains when no in-range
scores exist; counts remain zero and percentages/densities are `None`.

## Example and scale

Run [the simulated example](../examples/external_validation.py) with
`python examples/external_validation.py`. It includes three months, missing and
unexpected keys, duplicate keys, missing scores and a visible upward score shift.
The README shows both default and explicit baseline calls.

Joins, key counts, date derivation, quantiles, score-band assignment and
aggregations execute in Spark SQL/DataFrame APIs, compatible with Spark 3.5+.
Only scalar summaries, monthly aggregates, band aggregates and ten histogram
bins reach the driver; no raw rows, RDDs, SparkContext or `toPandas()` are used.
Driver memory scales with the number of months, not customer rows. Multiple
Spark actions and shuffles are required; users may cache inputs when reused.
The function does not persist inputs or write local files.
