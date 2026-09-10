# BrainModelKit

BrainModelKit is an early-stage Python library for building, evaluating, and
analyzing models. Its package structure is ready for optional Pandas and
PySpark integrations without requiring either framework for core usage.

> [!WARNING]
> The public API is not stable yet. BrainModelKit is currently under active
> development.

## Requirements

- Python 3.10 or newer

## Installation

Install the core package from PyPI:

```bash
python -m pip install BrainModelKit
```

Install an optional dataframe integration from PyPI:

```bash
python -m pip install "BrainModelKit[pandas]"
python -m pip install "BrainModelKit[pyspark]"
```

Install both integrations:

```bash
python -m pip install "BrainModelKit[dataframes]"
```

To use the unreleased source from a local clone, replace `BrainModelKit` with
`.` in the commands above, for example:

```bash
python -m pip install ".[pyspark]"
```

## Usage

```python
import brainmodelkit

print(brainmodelkit.__version__)
```

Integration-specific functionality lives under explicit namespaces:

```python
from brainmodelkit import pandas, pyspark
```

## Simulate data

### PySpark credit data

In a Jupyter notebook, install the package into the active kernel and restart
the kernel if prompted:

```python
%pip install "BrainModelKit[pyspark]"
```

Generate 10,000 rows with the default 20 numeric features:

```python
from brainmodelkit.pyspark import simulate_credit_data

credit_df, feature_columns = simulate_credit_data()

credit_df.show(10, truncate=False)
print(feature_columns)
```

You can pass an existing Spark session and change the dataset dimensions. This
example creates 5,000 rows and 30 numeric features:

```python
from pyspark.sql import SparkSession

from brainmodelkit.pyspark import simulate_credit_data

spark = SparkSession.builder.appName("CreditModelNotebook").getOrCreate()
credit_df, feature_columns = simulate_credit_data(
    spark,
    row_count=5_000,
    feature_count=30,
    max_days=30,
    seed=42,
)
```

The returned DataFrame contains `company_id`, `reference_date`, `status`,
`default_flag`, `industry_section`, `company_size`, and the requested
`feature_XX` columns. Company size is one of `Small`, `Medium`, `Large`, or
`Very Large` and contributes to the simulated default risk.
Use `default_flag` as the binary training label. `feature_columns` contains the
numeric input column names for a Spark ML `VectorAssembler`:

```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
model_data = assembler.transform(credit_df).select(
    "features",
    "default_flag",
)
train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)
```

## Metrics

### Spark

Install `BrainModelKit[pyspark]`. The examples below use a Spark DataFrame
named `spark_df` with `score`, `target`, and `segment` columns.

#### Kolmogorov-Smirnov (KS)

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

#### ROC AUC and Gini

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

#### Plot a ROC curve

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

#### Risk sorting table

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

### Pandas

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

#### Kolmogorov-Smirnov (KS)

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

#### ROC AUC and Gini

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

#### Plot a ROC curve

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

#### Risk sorting table

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

## Cube analysis

### Spark

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

### Pandas

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

## Model training

Use `brainmodelkit.training.pyspark.train_model` or
`brainmodelkit.training.pandas.train_model` to train binary classifiers,
evaluate OOT KS/AUC/Gini, and optionally score another dataset. Both accept
logistic regression, random forest, gradient boosting, LightGBM or a custom
estimator. Each run writes model information and feature importance to a unique
folder. MLflow model logging and signatures are optional.

### Spark

Install the integration from your local clone and configure Java/Spark:

```bash
python -m pip install -e ".[pyspark]"
```

Import `train_model` and pass your Spark DataFrames. This complete example
generates sample data and trains a random forest:

```python
from pyspark.sql import SparkSession

from brainmodelkit.pyspark import simulate_credit_data
from brainmodelkit.training.pyspark import train_model

spark = SparkSession.builder.appName("BrainModelTraining").getOrCreate()
data, features = simulate_credit_data(spark, row_count=1000, feature_count=6)
train_df, oot_df = data.randomSplit([0.75, 0.25], seed=42)

result = train_model(
    run_name="spark_random_forest",
    target_col="default_flag",
    feature_cols=features,
    train_df=train_df,
    oot_df=oot_df,
    model="random_forest",
    model_params={"numTrees": 20, "maxDepth": 5, "seed": 42},
    df_scoring=oot_df.drop("default_flag"),  # Optional, no target required.
    output_dir="training_runs",
    n_tiles=10,
)

print(result.metrics)  # oot_ks, oot_auc, oot_gini
print(result.output_dir)
result.scoring_predictions.select(*features, "score").show(5)

# The fitted pipeline includes feature assembly and accepts raw features.
result.model.transform(oot_df.drop("default_flag")).show(5)
```

Spark KS/AUC use tile approximations controlled by `n_tiles` (at least 2).
Use numeric, non-null features. Spark LightGBM additionally requires SynapseML
and its matching JVM packages configured in your Spark session.

To execute the standalone example from the repository root:

```bash
python examples/training/spark_example.py
```

### Pandas

Install the integration from your local clone:

```bash
python -m pip install -e ".[pandas]"
```

Import the Pandas version of `train_model` and pass Pandas DataFrames:

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from brainmodelkit.training.pandas import train_model

X, y = make_classification(n_samples=500, n_features=6, random_state=42)
features = [f"feature_{i}" for i in range(6)]
data = pd.DataFrame(X, columns=features).assign(target=y)
train_df, oot_df = train_test_split(
    data,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

result = train_model(
    run_name="pandas_random_forest",
    target_col="target",
    feature_cols=features,
    train_df=train_df,
    oot_df=oot_df,
    model="random_forest",
    model_params={"n_estimators": 50, "max_depth": 5, "random_state": 42},
    df_scoring=oot_df.drop(columns="target"),  # Optional, no target required.
    output_dir="training_runs",
)

print(result.metrics)  # Exact oot_ks, oot_auc, oot_gini
print(result.output_dir)
print(result.scoring_predictions.head())

# Reuse the fitted estimator with the same features in the same order.
predicted_labels = result.model.predict(oot_df[features])
```

To execute the standalone example from the repository root:

```bash
python examples/training/pandas_example.py
```

### Model options and outputs

Both backends accept `model="logistic_regression"` (default), `"random_forest"`,
`"gradient_boosting"`, or `"lightgbm"`. Pass algorithm parameters in
`model_params` using the backend's native names, as shown above. You may also
pass a custom estimator instance through `model`; it must support probability
predictions. For Pandas LightGBM, install `python -m pip install -e ".[pandas,lightgbm]"`.

Training labels must contain both classes, 0 and 1, and the target must not be
included in `feature_cols`. Preprocess your features before training. The examples
use random holdouts for demonstration; replace `oot_df` with a later-period
dataset for actual out-of-time evaluation.

`result.oot_predictions` and optional `result.scoring_predictions` include a
`score` column containing the probability of class 1. Each call creates a unique
subfolder under `output_dir` with `model_info.json` and `feature_importance.csv`.
These contain the model configuration, OOT metrics, and available native feature
importance or linear coefficients. The fitted model is returned as `result.model`;
the local report folder does not contain a serialized model.

For optional MLflow model persistence, install `python -m pip install -e ".[mlflow]"`,
configure your MLflow tracking URI and experiment, and add `mlflow_logging=True`
to either training call. Add `signature=True` to log the native input/output
signature (feature inputs and predicted labels). The MLflow run ID is available
as `result.run_id`.

See [runnable training examples and configuration](examples/training/README.md).

## Feature selection: RFE

Recursive feature elimination is a separate module. It trains the initial feature
set, removes up to `step` least important features, and refits until exactly
`n_feat_final` remain. The final subset is also trained and evaluated.

### Spark

Install `python -m pip install -e ".[pyspark]"` from your local clone and configure
Java/Spark. Using the Spark `train_df`, `oot_df`, and `features` from the training
example above:

```python
from brainmodelkit.feature_selection.pyspark import rfe

selection = rfe(
    run_name="spark_random_forest_rfe",
    target_col="default_flag",
    train_df=train_df,
    oot_df=oot_df,
    feature_cols=features,
    model="random_forest",
    model_params={"numTrees": 20, "maxDepth": 5, "seed": 42},
    step=2,
    n_feat_final=3,
    output_dir="feature_selection_runs",
    df_scoring=oot_df.drop("default_flag"),
    n_tiles=10,
)

print(selection.selected_features)
print(selection.history)
print(selection.output_dir)
print(selection.training_result.metrics)
selection.training_result.scoring_predictions.show(5)
# The final Spark pipeline assembles only the selected features.
selection.training_result.model.transform(oot_df).show(5)
```

Spark keeps input frames distributed and preserves caller-managed caches.
KS/AUC use the training module's tile approximations. LightGBM requires configured
SynapseML, including its matching JVM packages.

### Pandas

Install `python -m pip install -e ".[pandas]"` from your local clone. Using the
Pandas `train_df`, `oot_df`, and `features` from the training example above:

```python
from brainmodelkit.feature_selection.pandas import rfe

selection = rfe(
    run_name="pandas_random_forest_rfe",
    target_col="target",
    train_df=train_df,
    oot_df=oot_df,
    feature_cols=features,
    model="random_forest",
    model_params={"n_estimators": 50, "max_depth": 5, "random_state": 42},
    step=2,
    n_feat_final=3,
    output_dir="feature_selection_runs",
    df_scoring=oot_df.drop(columns="target"),
)

print(selection.selected_features)
print(selection.history)
print(selection.output_dir)
print(selection.training_result.metrics)
print(selection.training_result.scoring_predictions.head())
predicted_labels = selection.training_result.model.predict(
    oot_df[selection.selected_features]
)
```

### Supported models and reports

Both versions accept `logistic_regression`, `random_forest`, `gradient_boosting`,
and `lightgbm`, plus compatible custom estimators with native feature importance.
Pandas LightGBM needs `python -m pip install -e ".[pandas,lightgbm]"`.
Parameters use the same native names as `train_model`.

RFE ranks tree importance or absolute linear coefficients. Scale features before
coefficient-based selection when their units differ. Custom estimators must expose
one finite importance value per input feature; models without compatible importance
raise a clear error. Equal importance removes earlier input columns first, and
retained columns stay in their original order.

`step` and `n_feat_final` must be positive integers. The final count cannot exceed
the initial count; equal counts perform one fit. Omitting `feature_cols` selects
columns beginning with `feat_`; pass an explicit list for other naming conventions.
The target must not be a feature. OOT metrics are diagnostics and do not choose the
subset or stop elimination early. Only the final fit scores optional `df_scoring`.

Every call creates `feature_selection_runs/<unique-id>/` containing:

- `history.json`: each subset, removed features, OOT metrics, and training report path.
- `selected_features.json`: the final ordered feature list.
- One training report subfolder per iteration, with model information and importance.

`selection.training_result` is the final `TrainingResult`, including its fitted
model. Only this final model is retained in the returned result. For MLflow logging
of every fitted model, install the MLflow extra and pass `mlflow_logging=True`;
`signature=True` is optional. Local files contain reports, not serialized models.

Standalone examples are in [examples/feature_selection](examples/feature_selection/README.md).

## Development

Create a virtual environment and install the development dependencies:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

For local notebook development, install the small, reproducible notebook
dependency set:

```bash
python -m pip install -r requirements-notebooks.txt
```

Run the quality checks:

```bash
pytest
ruff check .
ruff format --check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete contribution guide.

## Project layout

```text
src/brainmodelkit/       Library source code
tests/                   Automated tests
pyproject.toml           Build and tool configuration
```

## License

BrainModelKit is distributed under the MIT License. See [LICENSE](LICENSE).
