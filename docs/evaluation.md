# Metrics and analysis cubes: examples and contracts

[Documentation home](README.md) ? [Modeling journeys](../README.md)

- [Spark evaluation module](#spark-evaluation-module)
- [Pandas evaluation module](#pandas-evaluation-module)
- [Metrics](#metrics)
- [Cube analysis](#cube-analysis)

## Spark evaluation module

Evaluate labeled predictions with `brainmodelkit.metrics.pyspark`;
use `brainmodelkit.cube_analysis.pyspark` for multiple scores and segment rollups.

| Function | Main parameters | Expected result |
| --- | --- | --- |
| `ks_ntile` | `dataframe`, `score_column`, `target`, `group_columns`, `n_tiles` | DataFrame with `KS` and optional group keys. |
| `auc_gini` | `df`, `score_column`, `target_column`, `group_by`, `n_tiles` | DataFrame with `auc`, `gini`, and optional group keys. |
| `curve_roc` | `df`, score/target column names, `n_tiles` | ROC points for plotting; not a fitted model. |
| `risk_table` | `df`, score/target column names, `n_tiles`, `ascending` | One row per occupied tile, including volume and event rate. |
| `calculate_metrics`, `calculate_ntile` | `df`, `group_columns`, `score_columns`, `target_column` | Metrics or risk tables for all requested dimension subsets. |

Higher scores should indicate target 1 for discrimination metrics. KS, AUC, and
Gini use decimal units. Spark metrics use tile approximations; see each function's execution notes.
See [cube output conventions](../docs/evaluation.md#output-conventions) for schemas and rollup labels.

### Spark evaluation metrics

The examples use this small labeled prediction table. Its scores perfectly rank
the two events above the two non-events, so overall KS, AUC, and Gini are all
`1.0` with the default tile setting. Real model outputs need not reach these values.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("EvaluationExample").getOrCreate()
spark_df = spark.createDataFrame(
    [(0.9, 1, "A"), (0.8, 1, "B"), (0.2, 0, "A"), (0.1, 0, "B")],
    ["score", "target", "segment"],
)
```

Install `BrainModelKit[pyspark]`. The examples below use a Spark DataFrame
named `spark_df` with `score`, `target`, and `segment` columns.

#### Spark Kolmogorov-Smirnov (KS)

Calculate approximate KS from score tiles using native Spark operations.
The default is 10 tiles; no Pandas conversion or Python UDF is used:

```python
from brainmodelkit.metrics.pyspark import ks_ntile

overall_ks = ks_ntile(dataframe=spark_df, n_tiles=10)
segment_ks = ks_ntile(
    dataframe=spark_df,
    group_columns="segment",
    n_tiles=20,
)

overall_ks.show()
segment_ks.show()
```

Defaults are `score_column="score"`, `target="target"`, and
`group_columns=None`. Results contain an uppercase `KS` column and any group
columns. KS is NaN when a dataset or group lacks either target class (0 or 1).
For custom columns, use
`ks_ntile(dataframe=spark_df, score_column="prediction", target="default_flag")`.
Existing positional calls continue to work.

#### Spark ROC AUC and Gini

```python
from brainmodelkit.metrics.pyspark import auc_gini

overall = auc_gini(df=spark_df, n_tiles=10)
by_segment = auc_gini(df=spark_df, group_by="segment", n_tiles=10)
overall.show()
```

Defaults are `score_column="score"`, `target_column="target"`, and `n_tiles=10`.
Use `score_column` and `target_column` for custom column names, and a list in
`group_by` for multiple grouping columns. Higher scores must indicate class 1.
Targets must be 0 or 1. Null/NaN pairs are removed, and empty or single-class
datasets/groups return NaN. Output columns are `auc` and `gini` plus group keys.
Gini is `2 * auc - 1`.

Spark AUC uses a score-tile approximation; it can differ from exact Pandas AUC.
Equal scores can span tiles, making results sensitive to their ordering.
Overall ranking uses an unpartitioned window. Grouped calculations collect
distinct group keys and compute each group separately, so use a modest number
of groups. The function runs Spark jobs immediately without collecting input rows.

#### Spark Plot a ROC curve

Install plotting dependencies with `python -m pip install pandas matplotlib`.
`curve_roc` returns a Spark DataFrame containing `tile`, `fpr` (false-positive
rate), and `tpr` (true-positive rate), including the origin at tile 0.

```python
import matplotlib.pyplot as plt
from brainmodelkit.metrics.pyspark import curve_roc

roc_df = curve_roc(df=spark_df, n_tiles=10)
points = roc_df.orderBy("tile").toPandas()
plt.plot(points["fpr"], points["tpr"], label="ROC")
plt.plot([0, 1], [0, 1], "--", label="Random classifier")
plt.xlabel("False-positive rate")
plt.ylabel("True-positive rate")
plt.legend()
plt.show()
```

Only the curve points (at most `n_tiles + 1`) are converted to Pandas.
This uses the same tile approximation as `auc_gini`. For a segment, pass
`df=spark_df.filter("segment = 'A'")`. Empty or single-class data return an
empty curve. Use `score_column` and `target_column` for custom column names.

#### Spark Risk sorting table

```python
from brainmodelkit.metrics.pyspark import risk_table

table = risk_table(df=spark_df, n_tiles=10)
table.show(truncate=False)

# Custom columns; use ascending=True when lower scores mean greater risk.
table = risk_table(
    df=spark_df,
    score_column="score",
    target_column="target",
    n_tiles=10,
    ascending=True,
)
```

The returned Spark DataFrame has one row per occupied tile, ordered by `n_tile`:

| Column | Meaning |
| --- | --- |
| `n_tile` | Tile number, starting at 1 |
| `minimum_range` | Lowest observed score in the tile (inclusive) |
| `maximum_range` | Highest observed score in the tile (inclusive) |
| `total_volume` | Number of rows with a valid score/target pair |
| `total_events` | Number of rows with target 1 |
| `total_non_events` | Number of rows with target 0 |
| `event_rate` | `total_events / total_volume`, between 0 and 1 |

Tile 1 contains the highest scores by default (`ascending=False`). Use
`ascending=True` if lower scores indicate greater risk. Tiles have nearly equal
row counts; extra rows go to earlier tiles. If fewer rows than `n_tiles` remain,
only occupied tiles are returned. `n_tiles=1` summarizes the entire input.
Null/NaN pairs are excluded, targets must be 0 or 1, and scores must be finite
numeric values. Empty input returns an empty table; single-class tiles have
an event rate of 0 or 1.

Ranges describe observed values, not reusable cutoff rules. Equal scores can
span tiles, so adjacent ranges may overlap. Spark ordering within tied scores
is unspecified, which can affect per-tile event counts. Filter `spark_df` first
to summarize a segment. Spark reuses score preparation and tiling; the table
uses an unpartitioned ranking window without collecting input rows. Calling
`risk_table` does not calculate KS, AUC, or ROC; calling KS does not build this table.

### Spark evaluation cubes

Install `BrainModelKit[pyspark]`. Start with overall metrics or an overall
risk table by n-tile, without segmentation:

```python
from brainmodelkit.cube_analysis.pyspark import calculate_metrics, calculate_ntile

cube = calculate_metrics(df=spark_df)
tile_cube = calculate_ntile(df=spark_df, n_tiles=10)
cube.show()
tile_cube.orderBy("score", "n_tile").show()
```

Optionally add grouping columns and multiple scores. Each cube includes every
subset of the grouping columns, including the overall result:

```python
cube = calculate_metrics(
    df=spark_df,
    group_columns=["segment", "period"],
    score_columns=["score_a", "score_b"],
    target_column="default_flag",
    n_tiles=10,
)
tile_cube = calculate_ntile(
    df=spark_df,
    group_columns=["segment", "period"],
    score_columns=["score_a", "score_b"],
    target_column="default_flag",
    n_tiles=10,
    ascending=True,
)
```

Spark uses the internal metric formulas with partitioned windows for grouped
AUC and risk tables. The cube collects neither group keys nor input rows and
uses no Pandas conversion or Python UDFs. Validation runs once per cube call
(one target check, plus one finite-score check for the n-tile cube); results
are evaluated when you run a Spark action. Each score still requires
`2 ** len(group_columns)` grouping calculations, so limit the number of
dimensions. Spark row order is unspecified; sort for display.
Spark metrics require `n_tiles >= 2`.

## Pandas evaluation module

Evaluate labeled predictions with `brainmodelkit.metrics.pandas`;
use `brainmodelkit.cube_analysis.pandas` for multiple scores and segment rollups.

| Function | Main parameters | Expected result |
| --- | --- | --- |
| `ks` | `dataframe`, `score_column`, `target`, `group_columns` | DataFrame with `KS` and optional group keys. |
| `auc_gini` | `df`, `score_column`, `target_column`, `group_by` | DataFrame with `auc`, `gini`, and optional group keys. |
| `curve_roc` | `df`, score/target column names | ROC points for plotting; not a fitted model. |
| `risk_table` | `df`, score/target column names, `n_tiles`, `ascending` | One row per occupied tile, including volume and event rate. |
| `calculate_metrics`, `calculate_ntile` | `df`, `group_columns`, `score_columns`, `target_column` | Metrics or risk tables for all requested dimension subsets. |

Higher scores should indicate target 1 for discrimination metrics. KS, AUC, and
Gini use decimal units. Pandas KS and AUC use exact calculations.
See [cube output conventions](../docs/evaluation.md#output-conventions) for schemas and rollup labels.

### Pandas evaluation metrics

Install `matplotlib` separately for plotting (`python -m pip install matplotlib`).

Install `BrainModelKit[pandas]` to include Pandas and scikit-learn.
Create a sample DataFrame for the examples below:

```python
import pandas as pd

df = pd.DataFrame(
    {
        "score": [0.9, 0.8, 0.2, 0.1],
        "target": [1, 1, 0, 0],
        "segment": ["A", "B", "A", "B"],
    }
)
```

#### Pandas Kolmogorov-Smirnov (KS)

Calculate exact point-by-point KS overall or by group:

```python
from brainmodelkit.metrics.pandas import ks

overall_ks = ks(dataframe=df)
segment_ks = ks(dataframe=df, group_columns="segment")
print(overall_ks)  # KS = 1.0
```

Defaults are `score_column="score"`, `target="target"`, and
`group_columns=None`. Results contain an uppercase `KS` column and any group
columns. KS is NaN when a dataset or group lacks either target class (0 or 1).
For custom columns, use
`ks(dataframe=df, score_column="prediction", target="default_flag")`.
Existing positional calls continue to work. The lower-level
`calculate_ks(y_true, y_pred)` function accepts two actual data series.

#### Pandas ROC AUC and Gini

```python
from brainmodelkit.metrics.pandas import auc_gini

overall = auc_gini(df=df)
by_segment = auc_gini(df=df, group_by="segment")
print(overall)  # auc = 1.0, gini = 1.0
```

Defaults are `score_column="score"`, `target_column="target"`, and `group_by=None`.
Results contain lowercase `auc` and `gini` columns, plus any grouping columns.
Use `score_column` and `target_column` for custom column names, and a list for
multiple grouping columns. Higher scores must indicate the positive class.
Missing target/score pairs are dropped; empty data or single-class groups
produce NaN. Gini is `2 * auc - 1`.

#### Pandas Plot a ROC curve

```python
import matplotlib.pyplot as plt
from brainmodelkit.metrics.pandas import curve_roc

roc_df = curve_roc(df=df)
plt.plot(roc_df["fpr"], roc_df["tpr"], label="ROC")
plt.plot([0, 1], [0, 1], "--", label="Random classifier")
plt.xlabel("False-positive rate")
plt.ylabel("True-positive rate")
plt.legend()
plt.show()
```

The returned Pandas DataFrame contains `fpr`, `tpr`, and `threshold`, in curve
order, including (0, 0) and (1, 1). The initial threshold is infinity.
Targets must be 0 or 1; null/NaN pairs are removed. Empty or single-class data
return an empty curve. For a segment, use `curve_roc(df=df[df["segment"] == "A"])`.
Use `score_column` and `target_column` for custom column names. Pandas computes
an exact curve; it does not use `n_tiles`. Plot each segment separately.

#### Pandas Risk sorting table

```python
from brainmodelkit.metrics.pandas import risk_table

table = risk_table(df=df, n_tiles=10)
print(table)

# Example with three tiles and ten observations.
example = pd.DataFrame({"score": range(1, 11), "target": [0] * 5 + [1] * 5})
print(risk_table(df=example, n_tiles=3))
```

| n_tile | minimum_range | maximum_range | total_volume | total_events | total_non_events | event_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 7 | 10 | 4 | 4 | 0 | 1.0 |
| 2 | 4 | 6 | 3 | 1 | 2 | 0.333333 |
| 3 | 1 | 3 | 3 | 0 | 3 | 0.0 |

The Pandas table uses the same columns, defaults, and tile sizes as Spark.
Set `score_column` and `target_column` for custom names, and `ascending=True`
when lower scores mean greater risk. Null/NaN pairs are excluded; targets must
be 0 or 1 and scores must be finite numeric values. Only occupied tiles are
returned, and empty data produce an empty table. The input DataFrame is unchanged.
Pandas preserves input order within tied scores. Ties may span tiles, so the
inclusive score ranges can overlap and tied results may differ from Spark.
For a segment, pass `df=df[df["segment"] == "A"]`. This function computes only
the requested risk summary, independently of KS, AUC, and ROC.

### Pandas evaluation cubes

Install `BrainModelKit[pandas]`. The same cube functions accept and return
Pandas DataFrames:

```python
from brainmodelkit.cube_analysis.pandas import calculate_metrics, calculate_ntile

cube = calculate_metrics(df=df)
tile_cube = calculate_ntile(df=df, n_tiles=10)
print(cube)
print(tile_cube)
```

Segmentation and custom score/target names are optional:

```python
cube = calculate_metrics(
    df=df,
    group_columns=["segment", "period"],
    score_columns=["score_a", "score_b"],
    target_column="default_flag",
)
tile_cube = calculate_ntile(
    df=df,
    group_columns=["segment", "period"],
    score_columns=["score_a", "score_b"],
    target_column="default_flag",
    n_tiles=10,
    ascending=True,
)
```

Pandas uses the internal exact KS/AUC metrics, so `calculate_metrics` has no
`n_tiles` parameter. Both backends default to `group_columns=None`,
`score_columns=("score",)`, and `target_column="target"`. A grouping or score
argument may be one column name or a sequence of names. Omitting
`group_columns`, passing `None`, or passing `[]` calculates overall results
only. `score_columns` must not be empty.

## Metrics

See [Spark evaluation](../docs/evaluation.md#spark-evaluation-module) or [Pandas evaluation](../docs/evaluation.md#pandas-evaluation-module) for metric examples.

## Cube analysis

Backend examples: [Spark cubes](../docs/evaluation.md#spark-evaluation-cubes) and [Pandas cubes](../docs/evaluation.md#pandas-evaluation-cubes).

### Output conventions

`calculate_metrics` returns grouping columns in the requested order followed
by `KS`, `AUC`, `Gini`, and `score`, with metrics rounded to five decimal places.
Higher scores should indicate target class 1. Missing pairs and single-class
groups follow the internal metric behavior. Empty input returns one overall
row per score with NaN metrics.

`calculate_ntile` returns grouping columns followed by `n_tile`,
`minimum_range`, `maximum_range`, `total_volume`, `total_events`,
`total_non_events`, `event_rate`, and `score`. Tiles are recalculated within
each group and rollup; they are not fixed global score bands. Tile 1 contains
the highest scores by default; use `ascending=True` for lowest scores first.
`n_tiles=1` summarizes each group. Only occupied tiles appear; empty input
returns an empty cube with the same columns. Missing pairs are excluded.
Ranges and event rates retain the internal risk table's precision. Tied scores
can span tiles, so ranges may overlap and backend results may differ for ties.

Both cubes cast group keys to strings, retain missing keys, and label omitted
dimensions `Geral`. A real key named `Geral` is indistinguishable from a rollup
in that column.

## Public alias and scalar utility

Both metric namespaces export `roc_auc_gini` as an alias of `auc_gini`; it accepts
the same arguments and returns the same DataFrame. The pandas namespace also
exports `calculate_ks`, which returns a scalar float from two pandas Series.
The underscore-prefixed helpers are implementation details, not public metrics.

Standalone pandas example (not dependent on the main workflow):

```python
import pandas as pd
from brainmodelkit.metrics.pandas import calculate_ks, roc_auc_gini

predictions = pd.DataFrame({"target": [1, 0, 1, 0], "score": [0.9, 0.1, 0.8, 0.2]})
print(calculate_ks(predictions.target, predictions.score))  # 1.0
print(roc_auc_gini(df=predictions))  # One row: auc=1.0, gini=1.0
```

Spark equivalent, continuing a caller-created Spark session:

```python
from brainmodelkit.metrics.pyspark import roc_auc_gini

predictions = spark.createDataFrame(
    [(1, 0.9), (0, 0.1), (1, 0.8), (0, 0.2)], ["target", "score"]
)
roc_auc_gini(df=predictions, n_tiles=10).show()  # auc=1.0, gini=1.0
```

Use binary 0/1 labels; higher scores mean class 1 for ROC AUC/Gini. Probabilities
are convenient but not required for ranking metrics. KS uses an absolute gap and
does not establish the score's direction. Pandas drops null pairs and returns NaN
when either class is absent; the scalar utility has no grouping argument. Align
Series indexes before passing them. Spark KS filters non-binary targets rather
than enforcing the AUC validator's contract; validate labels before evaluation.
Spark KS filters SQL null scores but does not explicitly remove floating NaN in
its input predicate. Clean NaN scores explicitly when calling it directly.
AUC/risk-table validation and missing-value handling are described in their
sections; do not assume all utilities reject invalid inputs identically.

## Interpreting segment results

Continuing either main workflow, pass `result.oot_predictions` to the matching
backend's cube functions. `calculate_metrics` returns one row per score and
observed group at every requested subtotal, including overall. `calculate_ntile`
adds `total_volume`, `total_events`, `total_non_events`, ranges, and event rates
per occupied score tile. It does not calculate KS/AUC/Gini.

**Illustrative values, not an executed benchmark:**

| segment | KS | AUC | Gini | score |
| --- | --- | --- | --- | --- |
| Geral | 0.40 | 0.75 | 0.50 | score |
| small | 0.25 | 0.65 | 0.30 | score |
| large | 0.50 | 0.80 | 0.60 | score |

The weaker `small` ranking deserves investigation even if the overall result
looks acceptable. Check tile volumes and class balance before interpreting the
gap; small groups have unstable estimates. No automatic minimum-volume threshold
is applied. Single-class metric groups yield NaN; their risk-table event rates
can legitimately be 0 or 1. Empty metric input retains an overall undefined row
per score; empty risk cubes have no occupied tiles.

With dimensions `segment` and `period`, rows cover both, each alone, and neither.
Omitted dimensions use `Geral`; actual missing keys remain missing. A real value
`Geral` is indistinguishable from a rollup label. Adding d dimensions causes
2**d grouping calculations per score. Spark cubes use native partitioned
calculations without collecting source rows or distinct group keys; pandas works
in memory. Metrics use uppercase `KS`, `AUC`, `Gini` and round to five decimals.
