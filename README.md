# BrainModelKit

BrainModelKit is an early-stage Python library for building, evaluating, and
analyzing models. Its base installation has no runtime dependencies; install
the execution stack that your application uses.

> [!WARNING]
> The public API is not stable yet. BrainModelKit is currently under active
> development and remains experimental, including its Spark integrations.

## Contents

Choose the Spark or Pandas chapter, then follow its modules in order. Shared
setup and detailed references follow both chapters.

- [Spark guide](#spark-guide)
  - [Spark simulated data](#spark-simulated-data)
    - [Spark credit simulation](#spark-credit-simulation)
    - [Spark history simulation](#spark-history-simulation)
  - [Spark features module](#spark-features-module)
    - [Create features with PySpark](#create-features-with-pyspark)
    - [Generate sample data, then create features](#generate-sample-data-then-create-features)
    - [Spark feature selection quickstart](#spark-feature-selection-quickstart)
  - [Spark training module](#spark-training-module)
    - [Spark training example](#spark-training-example)
      - [Choosing the LightGBM backend](#choosing-the-lightgbm-backend)
  - [Spark evaluation module](#spark-evaluation-module)
    - [Spark evaluation metrics](#spark-evaluation-metrics)
      - [Spark Kolmogorov-Smirnov (KS)](#spark-kolmogorov-smirnov-ks)
      - [Spark ROC AUC and Gini](#spark-roc-auc-and-gini)
      - [Spark Plot a ROC curve](#spark-plot-a-roc-curve)
      - [Spark Risk sorting table](#spark-risk-sorting-table)
    - [Spark evaluation cubes](#spark-evaluation-cubes)
  - [Spark validation and optimization](#spark-validation-and-optimization)
- [Pandas guide](#pandas-guide)
  - [Pandas simulated data](#pandas-simulated-data)
    - [Pandas history simulation](#pandas-history-simulation)
  - [Pandas features module](#pandas-features-module)
    - [Create features with Pandas](#create-features-with-pandas)
    - [Feature output summary](#feature-output-summary)
    - [Pandas feature selection quickstart](#pandas-feature-selection-quickstart)
  - [Pandas training module](#pandas-training-module)
    - [Pandas training example](#pandas-training-example)
  - [Pandas evaluation module](#pandas-evaluation-module)
    - [Pandas evaluation metrics](#pandas-evaluation-metrics)
      - [Pandas Kolmogorov-Smirnov (KS)](#pandas-kolmogorov-smirnov-ks)
      - [Pandas ROC AUC and Gini](#pandas-roc-auc-and-gini)
      - [Pandas Plot a ROC curve](#pandas-plot-a-roc-curve)
      - [Pandas Risk sorting table](#pandas-risk-sorting-table)
    - [Pandas evaluation cubes](#pandas-evaluation-cubes)
  - [Cross-validation](#cross-validation)
  - [Optimization](#optimization)
- [Shared setup and reference](#shared-setup-and-reference)
  - [Requirements](#requirements)
  - [Installation](#installation)
    - [SynapseML setup](#synapseml-setup)
  - [Environment Doctor](#environment-doctor)
  - [Usage](#usage)
  - [Simulate data](#simulate-data)
  - [Metrics](#metrics)
  - [Cube analysis](#cube-analysis)
    - [Output conventions](#output-conventions)
  - [Model Training](#model-training)
    - [Overview](#overview)
    - [Supported Models](#supported-models)
    - [Saving Trained Models](#saving-trained-models)
      - [Folder](#folder)
        - [How to execute `run_as="local"`](#how-to-execute-run_aslocal)
      - [MLflow](#mlflow)
        - [How to execute `run_as="mlflow"`](#how-to-execute-run_asmlflow)
        - [Databricks Serverless / Shared Compute](#databricks-serverless--shared-compute)
      - [No Persistence](#no-persistence)
        - [How to execute `run_as="none"`](#how-to-execute-run_asnone)
    - [Model Formats](#model-formats)
    - [Training Outputs](#training-outputs)
    - [Loading Trained Models](#loading-trained-models)
    - [Advanced Configuration](#advanced-configuration)
    - [Training parameter reference](#training-parameter-reference)
    - [Spark on Windows](#spark-on-windows)
    - [Migration and compatibility](#migration-and-compatibility)
  - [Feature selection](#feature-selection)
    - [Choose a method and read its algorithm](#choose-a-method-and-read-its-algorithm)
  - [Development](#development)
  - [Spark scalability limits](#spark-scalability-limits)
  - [Project layout](#project-layout)
  - [License](#license)
  - [Numeric time-series features](#numeric-time-series-features)
    - [Configure the generated features](#configure-the-generated-features)
  - [Workflow reading guide](#workflow-reading-guide)

## Spark guide

Install `brainmodelkit[pyspark]` and follow this chapter from simulation
through evaluation. See [Requirements](#requirements), [Installation](#installation),
and [Environment Doctor](#environment-doctor) before running the examples.
Each example states its own input setup; snippets are independent unless stated otherwise.

### Spark simulated data

Generate a synthetic credit table or a per-entity observation history.

Credit simulation accepts `spark_session`, `row_count` (10,000), `feature_count` (20),
`max_days` (30), and `seed` (42), plus `informative_fraction` and `signal_strength`
to control the feature signal. It returns `(credit_df, feature_columns)`.

#### Spark credit simulation

In a Jupyter notebook, install the package into the active kernel and restart
the kernel if prompted:

```python
%pip install "brainmodelkit[pyspark]"
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

By default, approximately one sixth of the numeric features contain a weak,
noisy target signal and the remaining features are independent noise. Control
this with `informative_fraction` and `signal_strength`; the signal is noisy and
does not directly copy the target, making the data useful for feature-selection
examples without creating trivially perfect predictors.

```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
model_data = assembler.transform(credit_df).select(
    "features",
    "default_flag",
)
train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)
```

#### Spark history simulation

`simulate_bureau_history` creates synthetic bounded score histories for the
supplied entity keys. This simulator is domain-specific; `build_features` can
also consume your own revenue, balances, limits, transactions, and other numeric
history. The history simulator does not generate a training target.

```python
from pyspark.sql import SparkSession

from brainmodelkit.pyspark import simulate_bureau_history

spark = SparkSession.builder.appName("HistoryExample").getOrCreate()
customers = spark.createDataFrame([("001",), ("002",)], ["account_id"])
history = simulate_bureau_history(
    customers,
    entity_cols="account_id",
    initial_date="2026-01-01",
    final_date="2026-03-31",
    frequency="monthly",
    seed=42,
    late_entry_fraction=0,
    missing_customer_fraction=0,
)
history.orderBy("account_id", "observation_date").show()
```

Expected result: six rows, one per account and month end (January 31,
February 28, March 31), with `account_id`, `observation_date`, and `bureau_score`.
The score is between 0 and 1000; its exact value depends on the seeded simulation.

| Parameters | Purpose and expected effect |
| --- | --- |
| `customers`, `entity_cols` | Input entities and one or more key columns; duplicate keys are deduplicated. |
| `initial_date`, `final_date`, `frequency` | Inclusive date range; `daily`, `weekly`, or `monthly` observations. |
| `seed` | Reproducible per-entity random histories. |
| `late_entry_fraction`, `missing_customer_fraction`, `missing_period_probability` | Control incomplete histories; zero cohort fractions give complete histories. |
| `score_min`, `score_max`, `persistence`, `noise_std` | Score bounds and evolution; defaults are 0, 1000, 0.85, and 25. |
| `observation_col`, `score_col` | Output column names; defaults are `observation_date` and `bureau_score`. |

See the [simulation contract](docs/temporal.md#simulation) for cohort rules and
calendar alignment.

### Spark features module

Use `brainmodelkit.pyspark.build_features` to enrich a spine from numeric history.
The result is a Spark DataFrame preserving the spine rows and original columns.
No generated lookup dates are returned: current and lag latency are numeric
`_date_diff_days` features measured against each spine reference date.

| Parameter | Meaning | Default / expected output |
| --- | --- | --- |
| `df_base`, `df_spine` | Historical observations, then rows to enrich. | Both use this backend's DataFrames. |
| `key_cols`, `value_cols` | Entity keys and numeric measures. | Required; single names or sequences. |
| `date_col`, `reference_date_col` | Observation and reference dates. | `observation_date`, `dt_ref`. |
| `period_unit` | Calendar units for windows and lags. | `months`; also `days` and `weeks`. |
| `windows` | Rolling window lengths, excluding the lower boundary. | `(3, 6, 9)`; statistics and current-minus-mean per window. |
| `lags` | Offsets for latest non-null as-of lookups. | `(1, 2, 8)`; value and latency per lag, plus current value and latency. |
| `aggregations` | Window statistics to calculate. | `mean`, `median`, `max`, `min`, `std`, `sum`, `count`. |
| `max_staleness_days` | Maximum age measured from each lookup cutoff. | `None`; does not filter windows. |

See [feature configuration](#configure-the-generated-features) for nulls,
validation, and legacy API compatibility. Selection then chooses which numeric
columns to send to training; it does not create the temporal features.

#### Create features with PySpark

Install `brainmodelkit[pyspark]` and use Spark DataFrames with the same arguments.
This standalone example requires a configured Spark/Java environment:

```python
from pyspark.sql import SparkSession, functions as F
from brainmodelkit.pyspark import build_features

spark = SparkSession.builder.appName("TemporalFeatures").getOrCreate()
df_base = spark.createDataFrame(
    [
        ("00000000000001", "2026-01-31", 600.0),
        ("00000000000001", "2026-02-28", 620.0),
        ("00000000000001", "2026-03-31", 650.0),
        ("00000000000001", "2026-04-30", 700.0),
    ],
    ["account_id", "observation_date", "revenue"],
).withColumn("observation_date", F.to_date("observation_date"))
df_spine = spark.createDataFrame(
    [("00000000000001", "2026-03-31")],
    ["account_id", "dt_ref"],
).withColumn("dt_ref", F.to_date("dt_ref"))

enriched = build_features(
    df_base,
    df_spine,
    key_cols=["account_id"],
    date_col="observation_date",
    reference_date_col="dt_ref",
    value_cols=["revenue"],
    period_unit="months",
    windows=[3],
    lags=[1],
    aggregations=["mean", "min", "max", "std", "count"],
)
enriched.show(truncate=False)
```

#### Generate sample data, then create features

This complete PySpark example connects `simulate_credit_data`,
`simulate_bureau_history`, and `build_features`. Install
`brainmodelkit[pyspark]` and configure Spark/Java first.

```python
from pyspark.sql import SparkSession, functions as F
from brainmodelkit.pyspark import (
    build_features,
    simulate_bureau_history,
    simulate_credit_data,
)

spark = SparkSession.builder.appName("GeneratedFeatureExample").getOrCreate()

# 1. Generate customers, a binary label, and three synthetic numeric inputs.
credit_df, original_features = simulate_credit_data(
    spark, row_count=20, feature_count=3, seed=42
)
# Fix the reference date so the example uses the same history window each run.
df_spine = credit_df.withColumn("reference_date", F.to_date(F.lit("2026-10-13")))

# 2. Generate monthly bureau scores for those same customers.
df_base = simulate_bureau_history(
    df_spine.select("company_id").distinct(),
    entity_cols="company_id",
    initial_date="2025-01-01",
    final_date="2026-10-13",
    frequency="monthly",
    seed=42,
    late_entry_fraction=0.0,
    missing_customer_fraction=0.0,
    observation_col="bureau_date",
    score_col="bureau_score",
)

# 3. Add features from the history to each row of the credit data.
enriched = build_features(
    df_base,
    df_spine,
    key_cols=["company_id"],
    date_col="bureau_date",
    reference_date_col="reference_date",
    value_cols=["bureau_score"],
    period_unit="months",
    windows=[3, 6, 9],
    lags=[1, 2, 8],
    aggregations=["mean", "min", "max", "std", "count"],
)

# 4. Inspect features and collect their column names for later model training.
created_features = [
    name
    for name in enriched.columns
    if name not in df_spine.columns and not name.endswith("_date")
]
model_features = original_features + created_features
print(model_features)
enriched.select(
    "company_id",
    "reference_date",
    "default_flag",
    "bureau_score__current",
    "bureau_score__lag_1_months",
    "bureau_score__mean_3_months",
    "bureau_score__count_3_months",
).orderBy("company_id").show(5, truncate=False)
```

`df_base` contains 21 month-end observations per customer, ending on September
30, 2026. `enriched` preserves the 20 credit rows, their original features, and
`default_flag`, and adds 26 numeric history features, including four latency
indicators. The three-month count is `3` for every row. The current score comes
from September 30; the one-month lag uses August 31 because its cutoff is
September 13. Setting both missing-history fractions to zero keeps this example
complete; real histories can produce null features.

Use `model_features` as `feature_cols` and `default_flag` as `target_col` when
training on separate train/OOT splits. Original spine dates and identifiers are
excluded from that numeric feature list; generated latency indicators are included. These synthetic histories demonstrate
the API; they are not designed to predict the simulated credit label.

#### Spark feature selection quickstart

Install `python -m pip install "BrainModelKit[pyspark]"` and configure Java.
This example builds data in Spark and runs RFECV with explicit binary-balanced
folds. It intentionally has a perfect synthetic signal to make the mechanics easy
to inspect; it is not a performance benchmark.

```python
from pyspark.sql import SparkSession, functions as F
from brainmodelkit.feature_selection import pyspark as fs

spark = SparkSession.builder.master("local[2]").appName("selection-demo").getOrCreate()
try:
    train = (
        spark.range(240)
        .withColumn("target", (F.col("id") % 2).cast("int"))
        .withColumn("signal", F.col("target").cast("double"))
        .withColumn("constant", F.lit(1.0))
        .withColumn("fold", (F.floor(F.col("id") / 2) % 3).cast("int"))
    )
    result = fs.rfecv(
        train,
        "target",
        ["signal", "constant"],
        model="decision_tree",
        cv=3,
        fold_col="fold",
    )
    print(result.selected_features)  # ["signal"]
    print(result.cv_results)
finally:
    spark.stop()
```

Spark selectors keep source rows distributed. The new permutation and Boruta
methods require classic Spark Python workers and use distributed sorting/indexing;
they do not support Spark Connect. MI uses binned counts in Spark, while Pandas
uses sklearn estimation. Read the
[Spark algorithm and execution guide](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/spark.md)
for fold semantics, bootstrap differences, model schemas, and compute costs.

### Spark training module

`brainmodelkit.training.pyspark.train_model` fits a model, evaluates OOT rows,
and optionally scores unlabeled rows or saves artifacts. Pass raw numeric feature
columns and keep the evaluation data separate from training.

| Parameters | Meaning / expected result |
| --- | --- |
| `run_name`, `target_col`, `feature_cols` | Run identity, binary label, and ordered model inputs. |
| `train_df`, `oot_df` | Training data and held-out evaluation data for this backend. |
| `model`, `model_params` | Algorithm and its backend-specific estimator settings. |
| `df_scoring` | Optional additional rows; label not required. |
| `save_model_to`, `model_format`, `output_dir` | Persistence destination, folder format, and artifact directory; `save_model_to="none"` keeps the run in memory. |
| `analysis_cube` | Optional OOT breakdown columns for grouped diagnostics. |

Expected result: `TrainingResult` with the fitted `model`, scalar `metrics`
(`oot_ks`, `oot_auc`, `oot_gini`), `feature_importance`, `oot_predictions`, and
optional `scoring_predictions`. Prediction frames include `score`, the class-1
probability. See the [full parameter reference](#training-parameter-reference),
[training outputs](#training-outputs), and [persistence options](#saving-trained-models).

#### Spark training example

Follow the [PySpark installation requirements](#installation) first.

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
    save_model_to="none",  # First run: train/evaluate without model persistence.
    n_tiles=10,
)

print(result.metrics)  # oot_ks, oot_auc, oot_gini
print(result.output_dir)
result.scoring_predictions.select(*features, "score").show(5)

# The fitted pipeline includes feature assembly and accepts raw features.
result.model.transform(oot_df.drop("default_flag")).show(5)
```

##### Choosing the LightGBM backend

Pass `lgbm_backend` directly to `train_model()`, outside `model_params`.
`"default"` selects SparkLightGBM and is used when omitted; `"synapseml"`
explicitly selects SynapseML. See [Installation](#installation) and
[SynapseML setup](#synapseml-setup) for prerequisites.

Using the same data and features from the example above:

```python
# SparkLightGBM: native LightGBM parameter names.
result = train_model(
    run_name="spark_lightgbm",
    target_col="default_flag",
    feature_cols=features,
    train_df=train_df,
    oot_df=oot_df,
    model="lightgbm",
    lgbm_backend="default",
    model_params={"num_iterations": 100, "learning_rate": 0.05},
    save_model_to="none",
)
```

```python
# SynapseML: Spark parameter names.
result = train_model(
    run_name="synapseml_lightgbm",
    target_col="default_flag",
    feature_cols=features,
    train_df=train_df,
    oot_df=oot_df,
    model="lightgbm",
    lgbm_backend="synapseml",
    model_params={"numIterations": 100, "learningRate": 0.05},
    save_model_to="none",
)
```

The same selector works in PySpark `cross_validate()` and `optimize()`;
keep it outside `model_params` and optimization `search_space`. It does not
apply to pandas training. Missing dependencies do not switch the backend.

Spark KS/AUC use tile approximations controlled by `n_tiles` (at least 2).
Use numeric, non-null features. SparkLightGBM requires classic Spark;
Spark Connect is unsupported. Its local/single-worker mode uses driver memory;
set `num_workers>1` in `model_params` for distributed training. See
[backend runtime limitations](docs/optimization.md#spark-lightgbm) for worker
memory and barrier execution requirements.

The SparkLightGBM adapter supports raw-feature pipelines, class-1 probabilities,
and Spark folder/MLflow persistence. Loading requires BrainModelKit and the
selected LightGBM backend in the target environment. Configure persistence
before choosing `save_model_to="folder"` or `"mlflow"` with either backend;
see [Spark on Windows](#spark-on-windows).

To execute the standalone example from the repository root:

```bash
python examples/training/spark_example.py
```

### Spark evaluation module

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
See [cube output conventions](#output-conventions) for schemas and rollup labels.

#### Spark evaluation metrics

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

##### Spark Kolmogorov-Smirnov (KS)

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

##### Spark ROC AUC and Gini

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

##### Spark Plot a ROC curve

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

##### Spark Risk sorting table

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

#### Spark evaluation cubes

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

### Spark validation and optimization

Use the Spark namespaces `brainmodelkit.model_selection.pyspark` and
`brainmodelkit.optimization.pyspark` for CV and Optuna workflows. Their backend
parameters and fold requirements are described in the [CV guide](docs/model_selection.md)
and [optimization guide](docs/optimization.md). Keep final OOT evaluation separate
from model selection; read the [shared optimization guidance](#optimization).

## Pandas guide

Install `brainmodelkit[pandas]` and follow this chapter from simulation
through evaluation. See [Requirements](#requirements), [Installation](#installation),
and [Environment Doctor](#environment-doctor) before running the examples.
Each example states its own input setup; snippets are independent unless stated otherwise.

### Pandas simulated data

Generate per-entity observation history natively in pandas. The classification
examples in the selection and training sections also show `make_classification`
from scikit-learn for labeled synthetic data; `simulate_credit_data` is Spark-only.

#### Pandas history simulation

`simulate_bureau_history` creates synthetic bounded score histories for the
supplied entity keys. This simulator is domain-specific; `build_features` can
also consume your own revenue, balances, limits, transactions, and other numeric
history. The history simulator does not generate a training target.

```python
import pandas as pd

from brainmodelkit.pandas import simulate_bureau_history

customers = pd.DataFrame({"account_id": ["001", "002"]})
history = simulate_bureau_history(
    customers,
    entity_cols="account_id",
    initial_date="2026-01-01",
    final_date="2026-03-31",
    frequency="monthly",
    seed=42,
    late_entry_fraction=0,
    missing_customer_fraction=0,
)
print(history)
```

Expected result: six rows, one per account and month end (January 31,
February 28, March 31), with `account_id`, `observation_date`, and `bureau_score`.
The score is between 0 and 1000; its exact value depends on the seeded simulation.

| Parameters | Purpose and expected effect |
| --- | --- |
| `customers`, `entity_cols` | Input entities and one or more key columns; duplicate keys are deduplicated. |
| `initial_date`, `final_date`, `frequency` | Inclusive date range; `daily`, `weekly`, or `monthly` observations. |
| `seed` | Reproducible per-entity random histories. |
| `late_entry_fraction`, `missing_customer_fraction`, `missing_period_probability` | Control incomplete histories; zero cohort fractions give complete histories. |
| `score_min`, `score_max`, `persistence`, `noise_std` | Score bounds and evolution; defaults are 0, 1000, 0.85, and 25. |
| `observation_col`, `score_col` | Output column names; defaults are `observation_date` and `bureau_score`. |

See the [simulation contract](docs/temporal.md#simulation) for cohort rules and
calendar alignment.

### Pandas features module

Use `brainmodelkit.pandas.build_features` to enrich a spine from numeric history.
The result is a Pandas DataFrame preserving the spine rows and original columns.
No generated lookup dates are returned: current and lag latency are numeric
`_date_diff_days` features measured against each spine reference date.

| Parameter | Meaning | Default / expected output |
| --- | --- | --- |
| `df_base`, `df_spine` | Historical observations, then rows to enrich. | Both use this backend's DataFrames. |
| `key_cols`, `value_cols` | Entity keys and numeric measures. | Required; single names or sequences. |
| `date_col`, `reference_date_col` | Observation and reference dates. | `observation_date`, `dt_ref`. |
| `period_unit` | Calendar units for windows and lags. | `months`; also `days` and `weeks`. |
| `windows` | Rolling window lengths, excluding the lower boundary. | `(3, 6, 9)`; statistics and current-minus-mean per window. |
| `lags` | Offsets for latest non-null as-of lookups. | `(1, 2, 8)`; value and latency per lag, plus current value and latency. |
| `aggregations` | Window statistics to calculate. | `mean`, `median`, `max`, `min`, `std`, `sum`, `count`. |
| `max_staleness_days` | Maximum age measured from each lookup cutoff. | `None`; does not filter windows. |

See [feature configuration](#configure-the-generated-features) for nulls,
validation, and legacy API compatibility. Selection then chooses which numeric
columns to send to training; it does not create the temporal features.

#### Create features with Pandas

Install `brainmodelkit[pandas]`, then run this complete example. It creates
features for revenue and balance using all seven supported aggregations, two
windows, two lags, and a staleness limit. The printed summary includes every
generated column, including latency in days.

```python
import pandas as pd

from brainmodelkit.pandas import build_features

df_base = pd.DataFrame(
    {
        "account_id": ["00000000000001"] * 4,
        "observation_date": pd.to_datetime(
            [
                "2026-01-31",
                "2026-02-28",
                "2026-03-31",
                "2026-04-30",
            ]
        ),
        "revenue": [600.0, 620.0, 650.0, 700.0],
        "balance": [1000.0, None, 900.0, 1100.0],
    }
)
df_spine = pd.DataFrame(
    {
        "account_id": ["00000000000001"],
        "dt_ref": pd.to_datetime(["2026-03-31"]),
    }
)

enriched = build_features(
    df_base,
    df_spine,
    key_cols=["account_id"],
    date_col="observation_date",
    reference_date_col="dt_ref",
    value_cols=["revenue", "balance"],
    period_unit="months",
    windows=[1, 3],
    lags=[1, 2],
    aggregations=["mean", "median", "min", "max", "std", "sum", "count"],
    max_staleness_days=60,
)

# Display all generated outputs for this reference row, including lookup latency.
feature_columns = [column for column in enriched if column not in df_spine.columns]
feature_summary = (
    enriched.loc[0, feature_columns].rename_axis("feature").reset_index(name="value")
)
print(feature_summary.to_string(index=False))

# Keep only numeric features when preparing inputs for a model.
model_features = [name for name in feature_columns if not name.endswith("_date")]
```

#### Feature output summary

The example adds **44 numeric columns**, including six lookup-latency features.
Each source column gets the same set of features. `feature_summary` is a display
table built from the enriched DataFrame; `build_features` returns the original
spine rows and columns with features appended.

For revenue at the March 31 reference date, the outputs are:

| Output | Example value | Meaning |
| --- | --- | --- |
| `revenue__current` | `650` | Latest non-null value on or before March 31. |
| `revenue__current_date_diff_days` | `0` | Days from the current observation to the reference date. |
| `revenue__lag_1_months` | `620` | Latest non-null value on or before February 28. |
| `revenue__lag_1_months_date_diff_days` | `31` | Days from the lag observation to the reference date. |
| `revenue__lag_2_months` | `600` | Latest non-null value on or before January 31. |
| `revenue__lag_2_months_date_diff_days` | `59` | Days from the lag observation to the reference date. |
| `revenue__mean_3_months` | `623.3333` | Arithmetic mean of the three window observations. |
| `revenue__median_3_months` | `620` | Exact median. |
| `revenue__min_3_months` | `600` | Minimum value. |
| `revenue__max_3_months` | `650` | Maximum value. |
| `revenue__std_3_months` | `25.1661` | Sample standard deviation (`ddof=1`). |
| `revenue__sum_3_months` | `1870` | Sum of non-null values. |
| `revenue__count_3_months` | `3` | Number of non-null observations. |
| `revenue__current_minus_mean_3_months` | `26.6667` | Current value minus the window mean. |

The seven statistics and current-minus-mean difference also appear with the
`_1_months` suffix. That window is `(2026-02-28, 2026-03-31]`, so it contains
only `650`: count is `1`, sum/mean/median/min/max are `650`, standard deviation
is null, and current-minus-mean is `0`. April observations are excluded from
all features because they occur after the reference date.

Balance illustrates missing observations: its one-month lag uses January 31's
`1000` because February's value is null. Its one-month lag latency is therefore 59 days relative to March 31,
and that observation satisfies the 60-day staleness limit measured from the
February 28 cutoff. Its three-month count is `2`, mean is `950`, sum is `1900`,
and current-minus-mean is `-50`. Each numeric column is evaluated independently.

Choose a subset of `aggregations`, `windows`, or `lags` to reduce the outputs.
Current values and their latency indicators are always included; each requested window also
adds current-minus-mean even when `"mean"` is omitted from `aggregations`.
See [Configure the generated features](#configure-the-generated-features) for
units, composite keys, empty-window behavior, and validation rules.

#### Pandas feature selection quickstart

Install `python -m pip install "BrainModelKit[pandas]"`. This complete example
creates data, splits before selection, removes a constant, selects two features,
and evaluates on held-out rows without writing files.

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from brainmodelkit.feature_selection import pandas as fs
from brainmodelkit.training.pandas import train_model

x, y = make_classification(
    n_samples=400, n_features=4, n_informative=2, n_redundant=0, random_state=42
)
features = ["income", "balance", "age", "tenure"]
data = pd.DataFrame(x, columns=features).assign(constant=1.0, target=y)
train, test = train_test_split(data, test_size=0.25, stratify=y, random_state=42)

quality = fs.variance_filter(train, feature_cols=features + ["constant"])
selection = fs.feature_importance_selection(
    train,
    "target",
    quality.selected_features,
    model="random_forest",
    model_params={"n_estimators": 30},
    top_k=2,
)
print(pd.DataFrame(selection.feature_table))
print("Selected:", selection.selected_features)
print("Rejected:", selection.rejected_features)

if not selection.selected_features:
    raise ValueError("No feature survived; inspect diagnostics before training.")
final = train_model(
    run_name="selected_features_demo",
    target_col="target",
    feature_cols=selection.selected_features,
    train_df=train,
    oot_df=test,
    model="random_forest",
    model_params={"n_estimators": 30, "random_state": 42},
    save_model_to="none",
)
print(final.metrics)
```

The constant is rejected by the quality filter; the importance selector retains
two remaining features. Scores measure the fitted model's importance, not causal
effects. In real projects use a later-period or group-separated test set where
appropriate. Fit encoders and imputers on training rows only, and inside CV folds
when using CV-based selection.

### Pandas training module

`brainmodelkit.training.pandas.train_model` fits a model, evaluates OOT rows,
and optionally scores unlabeled rows or saves artifacts. Pass raw numeric feature
columns and keep the evaluation data separate from training.

| Parameters | Meaning / expected result |
| --- | --- |
| `run_name`, `target_col`, `feature_cols` | Run identity, binary label, and ordered model inputs. |
| `train_df`, `oot_df` | Training data and held-out evaluation data for this backend. |
| `model`, `model_params` | Algorithm and its backend-specific estimator settings. |
| `df_scoring` | Optional additional rows; label not required. |
| `save_model_to`, `model_format`, `output_dir` | Persistence destination, folder format, and artifact directory; `save_model_to="none"` keeps the run in memory. |
| `analysis_cube` | Optional OOT breakdown columns for grouped diagnostics. |

Expected result: `TrainingResult` with the fitted `model`, scalar `metrics`
(`oot_ks`, `oot_auc`, `oot_gini`), `feature_importance`, `oot_predictions`, and
optional `scoring_predictions`. Prediction frames include `score`, the class-1
probability. See the [full parameter reference](#training-parameter-reference),
[training outputs](#training-outputs), and [persistence options](#saving-trained-models).

#### Pandas training example

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
    save_model_to="folder",
    model_format="pickle",
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

### Pandas evaluation module

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
See [cube output conventions](#output-conventions) for schemas and rollup labels.

#### Pandas evaluation metrics

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

##### Pandas Kolmogorov-Smirnov (KS)

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

##### Pandas ROC AUC and Gini

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

##### Pandas Plot a ROC curve

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

##### Pandas Risk sorting table

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

#### Pandas evaluation cubes

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

### Cross-validation

Use centralized CV on development data; it returns fold metrics and a summary.
Keep OOT data separate from fold creation and model selection.

```python
from brainmodelkit.model_selection.pandas import cross_validate

result = cross_validate(train_df, "target", features, "logistic_regression", cv=5)
print(result.fold_metrics)
print(result.summary)
```

See the [cross-validation guide](docs/model_selection.md) for K-Fold,
stratification, groups, time ordering, Spark folds, and leakage guidance.

### Optimization

Optuna optimization uses CV on `train_df`; OOT is reported separately and never
drives the default objective. Install the optional dependency with
`pip install "BrainModelKit[optimization]"`.

```python
from brainmodelkit.optimization.pandas import optimize

result = optimize(
    train_df,
    oot_df,
    "target",
    features,
    "logistic_regression",
    {"C": {"type": "float", "low": 0.01, "high": 10.0, "log": True}},
    n_trials=10,
)
print(result.best_params, result.best_value)
print(result.trials)
```


Constraints are optional and guide Optuna sampling throughout optimization:

```python
result = optimize(
    train_df,
    oot_df,
    "target",
    features,
    "logistic_regression",
    {"C": {"type": "float", "low": 0.01, "high": 10.0, "log": True}},
    objective_metric="oot_ks",
    direction="maximize",
    target_metric="train_oot_ks_abs_shift",
    target_limit=0.05,
    target_operator="<=",
)
```

To minimize instability subject to OOT KS >= 40%, use
`objective_metric="train_oot_ks_abs_shift", direction="minimize",
 target_metric="oot_ks", target_limit=0.40, target_operator=">="`.
All metrics use decimal units. Feasible trials outrank infeasible trials;
if no trial qualifies, the result explicitly reports failure and selects the
least-violating trial. Train, CV, OOT, and signed/absolute shift metrics are
available for AUC, Gini, and KS; unqualified names remain CV aliases.

**OOT used for optimization is a tuning validation set. Keep a separate untouched
temporal holdout for final evaluation.** See the [optimization guide](docs/optimization.md)
for metric names, selection semantics, diagnostics, and Spark backend setup.

## Shared setup and reference

Backend-independent setup, contracts, persistence, and project information.
The existing reference material is retained below.

### Requirements

- Python 3.10 or newer

The Python requirement is a packaging minimum, not a guarantee for every Spark
or optional-integration combination. See the [compatibility and release checks](docs/release-readiness.md)
for the explicit CI matrix, tested scope, and unverified environments.

### Installation

Install the execution stack you need from PyPI:

```bash
# Pandas
python -m pip install "brainmodelkit[pandas]"

# PySpark
python -m pip install "brainmodelkit[pyspark]"

# Pandas + MLflow
python -m pip install "brainmodelkit[pandas,mlflow]"

# PySpark + MLflow
python -m pip install "brainmodelkit[pyspark,mlflow]"
```

`brainmodelkit` by itself installs no execution backend. MLflow is independent
of both backends, so add `[mlflow]` only when model logging or MLflow model
persistence is needed. Persistence formats, ONNX conversion, Boruta, and
Optuna are also separate optional features; see the relevant sections below.

The `[pandas]` extra includes native LightGBM. `[pyspark]` includes SparkLightGBM
(`>=0.9.2,<0.10`); SynapseML is an explicit alternative via `[synapseml]`.
`[full]` is a convenience bundle, not every optional integration: Boruta, skops,
and ONNX still require `[boruta]`, `[secure]`, and `[onnx]`, respectively.

Spark requires a Java installation and a configured Spark environment. The
`[pyspark]` extra requires PySpark 3.5 or newer and includes SparkLightGBM.
For managed Spark, install `[sparklightgbm]` without replacing platform PySpark.
See [Choosing the LightGBM backend](#choosing-the-lightgbm-backend) for usage.

For a managed environment such as Databricks, use the platform-provided Spark
and MLflow versions when they meet those requirements:

```bash
python -m pip install --no-deps brainmodelkit
```

When installing directly from the repository, use the equivalent command:

```bash
python -m pip install --no-deps "git+https://github.com/Celso-RQ-Valle/BrainModelKit.git"
```

`--no-deps` prevents pip from installing, upgrading, or downgrading runtime
dependencies. Use it only after confirming that the platform already provides
the Pandas, PySpark, selected LightGBM backend, and other optional dependencies required by
the features you will call. In particular, do not use the PySpark extra merely
to replace a managed Spark runtime.

Development tools are available through `BrainModelKit[dev]`. Notebook and
plotting tools such as Jupyter and matplotlib are installed separately.

To use the unreleased source from a local clone, replace the package name with
`.` in the commands above, for example:

```bash
python -m pip install -e ".[pandas]"
```

#### SynapseML setup

For SynapseML, install the Python wrapper and load matching JVM artifacts
before the Spark session starts:

```bash
python -m pip install "brainmodelkit[synapseml]"
```

For Spark 4.0.x:

```python
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("BrainModelLightGBM")
    .config(
        "spark.jars.repositories",
        "https://mmlspark.blob.core.windows.net/maven",
    )
    .config(
        "spark.jars.packages",
        "com.microsoft.azure:synapseml_2.13:1.1.3-spark4.0",
    )
    .getOrCreate()
)
```

For Spark 3.5.x, use the compatible SynapseML `_2.12` coordinate documented by
[SynapseML](https://microsoft.github.io/SynapseML/docs/Get%20Started/InstallSynapseML/).
If the Python wrapper imports but the JVM package is missing, training raises
an actionable error instead of the opaque `JavaPackage object is not callable`.

### Environment Doctor

Inspect existing capabilities without changing the environment:

```python
import brainmodelkit as bmk

bmk.doctor()
bmk.doctor(backend="pandas")
bmk.doctor(backend="pyspark")
bmk.doctor(
    backend="pyspark",
    model="lightgbm",
    lgbm_backend="synapseml",
    save_model_to="mlflow",
)
bmk.doctor(verbose=True)
```

Doctor prints a report and returns a dictionary with capabilities, the result,
blocking capabilities, and recommendations. It recommends an available path
based on detected prerequisites, without assuming your workload size. Specify
`backend` when checking a model or persistence destination. Omitted persistence
checks no destination; training itself still defaults to folder persistence.

- `READY`: inspected prerequisites are present and meet package requirements.
- `PARTIAL`: some prerequisites exist, but a runtime check is incomplete or denied.
- `NOT AVAILABLE`: a required dependency is missing.
- `INCOMPATIBLE`: a detected version is below the package's minimum requirement.

Missing optional features affect only workflows that request them. Doctor does
not install anything, change package versions or environment variables, write
test files, change Spark configuration, download JARs, or create/restart Spark.
Package versions come from metadata without importing optional libraries.

In a Databricks notebook with an existing Spark session:

```python
bmk.doctor(verbose=True)
bmk.doctor(backend="pyspark", model="lightgbm", save_model_to="folder", verbose=True)
```

Compatible platform-provided Spark and MLflow installations are preserved.
Doctor inspects existing session references and distinguishes the PySpark Python
package, actual Spark runtime, and SynapseML Python wrapper from the JVM class.
For SynapseML, an installed wrapper alone does not establish JVM readiness.
For SparkLightGBM, both the bridge and native LightGBM must be installed. Restricted JVM
access or Spark Connect produces a partial check, not a recommendation to
replace platform packages. SynapseML artifacts must match the existing
Spark/Scala runtime; meeting `pyspark>=3.5` alone does not certify every pairing.

`READY` is a prerequisite check, not a training or persistence smoke test.
Doctor cannot safely verify native LightGBM execution, credentials, MLflow
tracking/artifact access, or filesystem write permissions. Spark folder writes
remain `PARTIAL` because filesystem/Hadoop support is not exercised. Without
an accessible existing JVM, Java version and SynapseML JVM checks are incomplete;
Doctor only looks for a Java executable and never launches it.

### Usage

```python
import brainmodelkit

print(brainmodelkit.__version__)
```

Integration-specific functionality lives under explicit namespaces:

```python
from brainmodelkit import pandas, pyspark
```

### Simulate data

Choose [Spark simulation](#spark-simulated-data) or [Pandas simulation](#pandas-simulated-data).

### Metrics

See [Spark evaluation](#spark-evaluation-module) or [Pandas evaluation](#pandas-evaluation-module) for metric examples.

### Cube analysis

Backend examples: [Spark cubes](#spark-evaluation-cubes) and [Pandas cubes](#pandas-evaluation-cubes).

#### Output conventions

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

### Model Training

Start with the [Spark training module](#spark-training-module) or [Pandas training module](#pandas-training-module).

#### Overview

`train_model` provides a unified interface for training, evaluating, scoring,
and optionally persisting models. Choose the backend import for your DataFrames.

- `model`: which algorithm or estimator is trained.
- `save_model_to`: the persistence destination/mechanism:
  - `"folder"` (default): saves the fitted model and training artifacts in a run directory.
  - `"mlflow"`: logs the training run and model through MLflow.
  - `"none"`: keeps the trained model in memory without creating persistence artifacts.
- `model_format`: selects serialization for folder persistence only; ignored
  for MLflow and `"none"`.

The defaults are `save_model_to="folder"`, `model_format=None`, and
`output_dir="training_runs"`. For example, with Pandas data and LightGBM installed:

```python
from brainmodelkit.training.pandas import train_model

result = train_model(
    run_name="credit_model_v1",
    target_col="target",
    feature_cols=features,
    train_df=train_df,
    oot_df=oot_df,
    model="lightgbm",
)
```

This implicitly uses `save_model_to="folder"` and saves a pickle model.

Pass `analysis_cube=["size", "segment", "date"]` to calculate grouped KS,
AUC, and Gini diagnostics from the OOT predictions. The cube is optional and
is kept separate from the scalar OOT metrics:

```python
result = train_model(
    "credit_model_v1",
    "target",
    features,
    train_df,
    oot_df,
    analysis_cube=["size", "segment", "date"],
    save_model_to="none",
)
print(result.analysis_cube)
```

Pandas returns a Pandas DataFrame; PySpark returns a Spark DataFrame. Breakdown
columns must exist in the OOT frame and are never used to train the model.

#### Supported Models

Both backends accept `model="logistic_regression"` (default), `"random_forest"`,
`"gradient_boosting"`, or `"lightgbm"`. Pass algorithm parameters in
`model_params` using the backend's native names, as shown above. You may also
pass a custom estimator instance through `model`; it must support probability
predictions. LightGBM is included in the Pandas extra.

Training labels must contain both classes, 0 and 1, and the target must not be
included in `feature_cols`. Preprocess your features before training. The examples
use random holdouts for demonstration; replace `oot_df` with a later-period
dataset for actual out-of-time evaluation.

#### Saving Trained Models

The calls below use the Pandas feature selection quickstart's data and `target` label. For the
Spark feature selection quickstart, use its import and replace `"target"` with `"default_flag"`.
Choose one mode:

| `save_model_to` | Equivalent deprecated alias | Setup | Result location |
| --- | --- | --- | --- |
| `"folder"` (default) | `run_as="local"` | Writable folder; configured Hadoop for Spark writes | `output_dir`, `model_uri` |
| `"mlflow"` | `run_as="mlflow"` | MLflow extra and tracking/artifact access; Hadoop for Spark writes | `run_id`, `model_uri` |
| `"none"` | `run_as="none"` | Training backend only | All locations and run ID are `None` |

Options specific to another destination are ignored. In particular, MLflow
does not use `output_dir`, `model_format`, `save_path`, `save_format`,
`save_metadata` or Spark `overwrite`. `signature` is used only by MLflow.
Contradictory destination arguments still raise an error.

##### Folder

###### How to execute `run_as="local"`

Use `save_model_to="folder"` in the example below (recommended), or replace it
with `run_as="local"` in an existing notebook. Both perform the same operation.
Relative paths start at your script/notebook's working directory. `output_dir`
is a parent directory, not a model filename. Each execution gets a new run ID,
so rerunning does not replace the previous run. Read `result.model_uri` to
locate the actual saved model.

`save_model_to="folder"` saves under `output_dir` in a unique directory named
`<sanitized_run_name>_<run_id>`. The default root is `training_runs`.
Each run includes the model, `model_info.json`, `metrics.json`,
`feature_importance.csv`, and `metadata.json`.

```python
result = train_model(
    "credit_folder",
    "target",
    features,
    train_df,
    oot_df,
    save_model_to="folder",
    output_dir="training_runs",
)
print(result.output_dir, result.model_uri)
```

##### MLflow

###### How to execute `run_as="mlflow"`

Use `save_model_to="mlflow"` below, or replace it with `run_as="mlflow"`.
First install the dependencies and configure tracking. For a remote server,
use `mlflow.set_tracking_uri("http://localhost:5000")` with your actual server
address and credentials. The example below uses local SQLite tracking.
Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` in the same working
directory to inspect it, then open the URL printed by MLflow. SQLite stores
metadata; model files go to the experiment's artifact location.

Install the MLflow extra together with the backend you use:

```bash
python -m pip install "BrainModelKit[pandas,mlflow]"
# or: python -m pip install "BrainModelKit[pyspark,mlflow]"
```

Configure a tracking URI and experiment in a writable working directory:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("brainmodelkit-training")
result = train_model(
    "credit_tracked",
    "target",
    features,
    train_df,
    oot_df,
    save_model_to="mlflow",
    signature=True,
)
```

Both backends log their native MLflow flavor, parameters, finite metrics,
metadata tags, and the four report files under `training_report/`.
An existing active run causes a nested run. Temporary reports are removed after
upload. `result.model_uri` is `runs:/<run_id>/model`; `result.output_dir` is `None`.

###### Databricks Serverless / Shared Compute

Normal usage remains unchanged for local environments, Colab, and Spark setups
where MLflow already works without a DFS temporary directory:

```python
result = train_model(
    "experiment",
    "target",
    features,
    train_df,
    oot_df,
    save_model_to="mlflow",
)
```

Databricks Serverless/shared compute may require a Unity Catalog Volume while
MLflow serializes Spark ML models. When MLflow reports that requirement, use
the Spark `train_model` parameter:

```python
result = train_model(
    "experiment",
    "target",
    features,
    train_df,
    oot_df,
    save_model_to="mlflow",
    mlflow_dfs_tmp="/Volumes/<catalog>/<schema>/<volume>/mlflow_tmp",
)
```

Alternatively, configure
`MLFLOW_DFS_TMP=/Volumes/<catalog>/<schema>/<volume>/mlflow_tmp` in your environment.
The optional parameter defaults to `None`, which leaves MLflow's existing
defaults/environment handling unchanged. An explicit value is passed only to
`mlflow.spark.log_model` as `dfs_tmpdir`; generic DFS paths are accepted without
requiring `/Volumes/`. This does not apply to Pandas/sklearn MLflow persistence.
BrainModelKit neither creates Unity Catalog volumes nor changes your storage,
Spark configuration, or environment variables.

##### No Persistence

###### How to execute `run_as="none"`

Use `save_model_to="none"` below, or replace it with `run_as="none"`.
Inspect `result.metrics` and use `result.model` for subsequent predictions.
The model exists only in the current session. All persistence options are
ignored, including `signature=True`; this mode does not import MLflow.

```python
result = train_model(
    "experiment",
    "target",
    features,
    train_df,
    oot_df,
    save_model_to="none",
)
```

Training, evaluation, and optional scoring still run. No directories, model
files, or report artifacts are created, and no MLflow run is started or logged.
`result.output_dir`, `result.model_uri`, and `result.run_id` are `None`.

#### Model Formats

For folder persistence, `model_format=None` resolves to `"pickle"` for Pandas
and `"spark"` for PySpark. Pandas also supports `joblib`, `cloudpickle`, `skops`,
`onnx`, and `native`. Spark supports its native pipeline format only.

```python
result = train_model(
    "credit_joblib",
    "target",
    features,
    train_df,
    oot_df,
    save_model_to="folder",
    model_format="joblib",
)
```

Joblib, cloudpickle, skops, and ONNX are separate optional persistence extras;
they are imported only when the corresponding format is requested.
Native serialization supports LightGBM, XGBoost, and CatBoost. ONNX requires a
supported skl2onnx converter and uses one training row to describe inputs.
Spark persistence requires a configured Spark/Hadoop environment.

`model_format` is used and validated only for folder persistence. MLflow manages
its own serialization; `none` does not serialize. Both ignore folder options.

#### Training Outputs

`TrainingResult` exposes `model`, `metrics`, `feature_importance`,
`oot_predictions`, `scoring_predictions`, `save_model_to`, `model_format`,
`run_id`, `model_uri`, and `output_dir`. `model_format` is the resolved folder
format, or `None` for MLflow/no persistence. Prediction frames include `score`,
the probability of class 1; `scoring_predictions` is `None` without `df_scoring`.
Metrics are `oot_ks`, `oot_auc`, and `oot_gini`.

`model_info.json` records the model, run name/ID, target, features, and serializable
parameters. `metrics.json` stores metrics, with undefined values represented as
JSON null. `feature_importance.csv` contains native importances or signed
coefficients. `metadata.json` records versions, timestamp, destination, format,
framework, and model class. Persistence calculates no additional metrics.

#### Loading Trained Models

```python
from brainmodelkit.persistence import load_model

model = load_model(result.model_uri, format="joblib")  # Match the saved format.
```

Use `format="mlflow"` for MLflow URIs. For Spark, pass `format="spark"` and
`model_class=pyspark.ml.PipelineModel`. Native models require their loader class;
LightGBM files use `model_class=lightgbm.Booster`, including sklearn wrapper saves.
ONNX loads as an `onnxruntime.InferenceSession` with its `run` API.
Formats are never guessed from filename extensions. Only load trusted artifacts
with pickle, joblib, cloudpickle, or MLflow; reviewed skops types can be passed
through `trusted=[...]`.

#### Advanced Configuration

Pass backend-specific estimator options through `model_params` and optional
unlabeled data through `df_scoring`. Pandas custom estimators must be cloneable
and implement `predict_proba`; Spark estimators must expose standard probabilistic
classifier columns. Inputs are not modified.

`signature=True` applies only to MLflow and describes native prediction labels, not the
added score column. Spark `n_tiles` controls metric approximation and must be at
least 2. Spark `overwrite` is forwarded to its native writer; automatic run
directories remain unique.

#### Training parameter reference

Both backends accept the following arguments unless marked Spark-only.
Arguments after `model` must be passed by keyword.

| Parameter | Default | Meaning and usage |
| --- | --- | --- |
| `run_name` | Required | Human-readable name. Folder names are sanitized and receive a unique ID; MLflow uses this as the run name. |
| `target_col` | Required | Label column in training/OOT, with non-null 0/1 values. Training must contain both classes. Never include it in features. |
| `feature_cols` | Required | Nonempty sequence of unique column names, in model input order. Must exist in all supplied frames. Preprocess before training; Spark requires numeric, non-null inputs. |
| `train_df` | Required | Labeled development DataFrame used to fit. Its backend must match the import. |
| `oot_df` | Required | Separate labeled evaluation frame. Prefer a later period for real OOT. Single-class metrics are undefined (NaN). |
| `model` | `"logistic_regression"` | Name: `logistic_regression`, `random_forest`, `gradient_boosting`, `lightgbm`; or compatible estimator. A fresh clone/copy is fitted. |
| `model_params` | `None` | Estimator options using native backend names, e.g. Pandas `{"max_iter": 100}` versus Spark `{"maxIter": 100}`. `None` uses estimator defaults. |
| `lgbm_backend` (Spark-only) | `"default"` | LightGBM backend: `"default"` (SparkLightGBM) or `"synapseml"`. Pass directly to `train_model()`, outside `model_params`; see [backend examples](#choosing-the-lightgbm-backend). |
| `df_scoring` | `None` | Optional frame with every feature. Label is optional and may remain present. Omission produces `scoring_predictions=None`. |
| `save_model_to` | `"folder"` | `folder`, `mlflow` or `none`; see execution examples above. |
| `output_dir` | `"training_runs"` | Folder-mode parent directory, created if needed. Ignored in other modes, where `None` is accepted. |
| `model_format` | `None` | Folder format, defaulting to Pandas `pickle` or Spark `spark`. Other formats/dependencies are listed above. Ignored outside folder mode. |
| `signature` | `False` | Infer MLflow input/native prediction schemas. Does not change fitting or add score output to the logged model. Ignored outside MLflow. |
| `analysis_cube` | `None` | Grouping column or list, e.g. `["company_size", "industry_section"]`, present in OOT. Returns `result.analysis_cube`. `[]` requests overall metrics; `None` disables it. More dimensions create exponentially more groups. Not included in the four persisted reports. |
| `n_tiles` (Spark-only) | `10` | Integer >= 2 for approximate KS/AUC/Gini and the optional metric cube. Changes approximation granularity, not model fitting. Pandas has no training `n_tiles` argument. |
| `overwrite` (Spark-only) | `False` | Native folder-writer overwrite, useful with an explicit legacy `save_path`. Generated run directories remain unique. Ignored outside folder mode. |
| `run_as`, `runs_as` | `None` | Deprecated destination aliases: `local`, `mlflow`, `none`. Conflicting destinations remain errors. |
| `mlflow_logging` | `None` | Deprecated: `True` selects MLflow; `False` does not override the destination. |
| `save_path` | `None` | Deprecated explicit folder-mode model path. Reports still use `output_dir`. Ignored in other modes. |
| `save_format` | `None` | Deprecated folder format alias. Conflicting folder formats are errors. Ignored outside folder mode. |
| `save_metadata` | `True` | Sidecar for an explicit legacy folder `save_path` only. Does not disable standard reports or MLflow tags. |

Input `score` is reserved in both backends. Spark also reserves `features`,
`prediction`, `probability`, `rawPrediction`: pass raw feature columns rather
than an assembled frame. Predictions preserve input columns and add outputs.
Dropping the label from `df_scoring` is optional and does not fix saving errors.

#### Spark on Windows

Spark can fit successfully and fail when saving. Folder and MLflow destinations
both use Spark's native writer. MLflow may write directly to a remote artifact
filesystem or stage the Spark model through a Hadoop/local filesystem before
uploading it; changing `output_dir`, disabling
`signature`, or dropping the scoring label does not remove this dependency.
See [MLflow Spark persistence](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.spark.html).

An exception mentioning `Shell.checkHadoopHome`, `HADOOP_HOME` or `winutils.exe`
points to Windows Hadoop configuration. For folder persistence on a local
filesystem, BrainModelKit checks the Windows driver helper before fitting when
Hadoop NativeIO is unavailable. Remote filesystems, Linux drivers, and sessions
without a driver JVM do not run this local check. MLflow selects its own save
path without a winutils preflight; actual Hadoop save failures include recovery
instructions. MLflow local staging can still require Windows Hadoop support.
`save_model_to="none"` performs no persistence validation. These checks do not
guarantee filesystem permissions, native-library compatibility or artifact access.

1. Use `save_model_to="none"` to train/evaluate without model persistence while
   configuring the environment. The model remains in `result.model`.
2. For saving, configure Windows Hadoop binaries/native libraries compatible
   with Spark's bundled Hadoop. `HADOOP_HOME` must point to their root containing
   `bin/winutils.exe`; native libraries must be on the process path. Setting a
   variable alone does not install these files.
3. Set up the environment before starting Python/Spark and restart the notebook
   kernel/process. Hadoop caches initialization in the JVM.
4. Alternatively use a configured Linux/WSL Spark environment. A remote MLflow
   server alone does not change the Spark driver's operating system.

Inspect the active classic Spark JVM:

```python
print(spark.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion())
print(spark.sparkContext._jvm.java.lang.System.getProperty("os.name"))
```

See [Hadoop Shell](https://hadoop.apache.org/docs/current/api/org/apache/hadoop/util/Shell.html)
for helper resolution. MLflow database migration and experiment-creation INFO
messages are normal first-use output; inspect the final exception and its cause.

#### Migration and compatibility

| Previous API | New API |
| --- | --- |
| `run_as="local"` | `save_model_to="folder"` |
| `run_as="mlflow"` | `save_model_to="mlflow"` |
| `run_as="none"` | `save_model_to="none"` |
| `runs_as=...` | Deprecated alias; use the same mappings above |
| `result.run_as` | `result.save_model_to` (`"folder"` replaces `"local"`) |

Legacy aliases emit `DeprecationWarning`; incompatible destination arguments
raise `ValueError`. The deprecated result property retains historical values.
`mlflow_logging=True` remains a deprecated alias for MLflow. Replace `save_format`
with `model_format`; `save_format="mlflow"` raises a migration error in folder mode.
Replace `save_path` with `output_dir` and read `result.model_uri`. Deprecated
explicit paths still save the model at that path and reports in the run directory;
`save_metadata` controls only the external model sidecar. External paths take
effect only with folder persistence. Read metrics from `metrics.json`.

### Feature selection

Start here to choose features, inspect the reasons, and evaluate a final model.
All sixteen methods are available in both Pandas and Spark. Backend algorithms
and arguments can differ; each method link below explains the actual computation,
assumptions, decision rules, diagnostics, and examples.

Examples: [Spark feature selection quickstart](#spark-feature-selection-quickstart) and [Pandas feature selection quickstart](#pandas-feature-selection-quickstart).

#### Choose a method and read its algorithm

| Method | Question answered | Detailed guide (Pandas and Spark) |
| --- | --- | --- |
| `completeness` | Is the feature sufficiently available, globally and by group? | [Completeness](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/completeness.md) |
| `variance_filter` | Is there variation beyond a chosen floor? | [Variance](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/variance_filter.md) |
| `cardinality` | Does the distinct-value count fit the intended use? | [Cardinality](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/cardinality.md) |
| `correlation_filter` | Which numeric features are redundant with earlier retained ones? | [Correlation](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/correlation_filter.md) |
| `mutual_information` | Is there univariate dependence on the target? | [Mutual information](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/mutual_information.md) |
| `chi_square` | Do count/category features show target association? | [Chi-square](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/chi_square.md) |
| `anova` | Do numeric class means differ? | [ANOVA](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/anova.md) |
| `information_value` | How different are event/non-event bin distributions? | [IV and WoE](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/information_value.md) |
| `feature_importance_selection` | Which features have high native model importance? | [Importance](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/feature_importance_selection.md) |
| `l1_selection` | Which penalized logistic coefficients remain nonzero? | [L1](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/l1_selection.md) |
| `permutation_importance_selection` | How much does held-out performance drop after shuffling? | [Permutation](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/permutation_importance_selection.md) |
| `rfe` | Which features survive elimination to a fixed count? | [RFE](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/rfe.md) |
| `rfecv` | Which visited feature count maximizes CV performance? | [RFECV](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/rfecv.md) |
| `sequential_selection` | Which greedy additions/removals improve CV score? | [Sequential](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/sequential_selection.md) |
| `stability_selection` | Which features recur across resamples/groups? | [Stability](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/stability_selection.md) |
| `boruta` | Which real features consistently beat shuffled shadows? | [Boruta hypotheses and decisions](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/boruta.md) |

Pandas Boruta requires `BrainModelKit[pandas,boruta]`; Spark Boruta uses the
native forest with the Spark extra. Its guide explains shadow construction,
binomial hit tests, multiple-testing corrections, and why an unresolved
hypothesis is not a validated null.

Except for historical RFE (`RFEResult`), selectors return `SelectionResult`.
Inspect `feature_table` and carry `selected_features` into the next step.
Ranked methods do not invent a cutoff: set `top_k` or a threshold intentionally.
RFE writes iteration reports; the other selection calls in these examples do not
write models or start MLflow runs.

Read the
[full feature-selection guide](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection.md)
and
[runnable examples](https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/examples/feature_selection/README.md).
Documentation links are absolute so they also work when this README is rendered
on PyPI. New Spark methods described here are in the source tree; install the
checkout with `pip install -e ".[pyspark]"` until a release containing them is published.

### Development

Create a virtual environment and install the development dependencies:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Release CI explicitly adds optional integrations and rejects skipped tests:

```bash
python -m pip install -e ".[dev,pandas,optimization,mlflow,boruta,persistence]" "pyspark==4.0.2"
python -m pytest --fail-on-skip
```

Use a Python/Spark pair from the [CI matrix](docs/release-readiness.md).
Local tests without the extra dependencies can skip optional integrations;
Windows without Hadoop support can skip Spark disk round trips. Neither counts
as release validation. CI also builds the distributions and runs
`python scripts/check_wheel.py --spark-version 4.0.2` in fresh virtual environments.

For local notebook development, install the notebook
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

### Spark scalability limits

Spark execution does not establish universal big-data readiness. Current CI
uses small local datasets, not distributed load benchmarks. Standalone overall
metrics use unpartitioned ranking windows; grouped AUC collects distinct group
keys and runs calculations per group. Cube analysis avoids that group-key
collection, but computes `2 ** len(group_columns)` grouping subsets per score.

Feature selection can run many Spark actions and repeated model fits. Correlation
materializes a feature-by-feature matrix on the driver (default limit: 256
features); other selectors collect aggregate summaries. Temporal feature joins
can expand substantially with long histories and many reference dates; exact
median needs group-size memory. Bound feature counts, grouping dimensions, and
history batches, and measure your workload before scaling. Spark Connect and
managed clusters are not covered by release CI; SparkLightGBM requires classic
Spark. Its single-worker mode (including automatic local mode) materializes
training rows in driver NumPy memory; distributed native training requires
`num_workers>1`. See [validation scope and limitations](docs/release-readiness.md).

### Project layout

```text
src/brainmodelkit/       Library source code
tests/                   Automated tests
pyproject.toml           Build and tool configuration
```

### License

BrainModelKit is distributed under the MIT License. See [LICENSE](LICENSE).
### Numeric time-series features

Use `build_features(df_base, df_spine, ...)` to create current
values, lags, and rolling statistics from historical numeric data. Pass the
historical source **first** and the rows to enrich (the spine) **second**:

- `df_base` contains entity keys, observation dates, and numeric values.
- `df_spine` contains entity keys and a reference date for each row to enrich.
  Its original rows and columns are preserved.

#### Configure the generated features

Pass multiple numeric columns in `value_cols` to generate features for each,
or multiple keys in `key_cols` for composite entities. `period_unit` accepts
`"days"`, `"weeks"`, or `"months"`. Defaults are `windows=[3, 6, 9]` and
`lags=[1, 2, 8]`; supported aggregations are `mean`, `median`, `max`, `min`,
`std`, `sum`, and `count`.

Current and lag features use the latest non-null value on or before their
cutoff, rather than requiring an observation on that exact date. Rolling
windows include dates in `(reference date - window, reference date]`.
Lookup latency columns end in `_date_diff_days` and measure calendar days from
the source observation to the spine reference date, including for lags. Missing
lookups have null latency. Generated lookup dates are not returned; original
spine dates remain unchanged. The legacy APIs retain their date outputs.
Set `max_staleness_days=60`, for example, to reject current/lag observations
more than 60 days before their respective cutoffs; this does not filter windows.

Keys and dates must be non-null, and history must have at most one row per
entity and observation date. Missing lookups return null; empty windows have
zero counts and null statistics. Handle missing numeric features before passing
them to an estimator that requires complete inputs.

For synthetic input histories, use `simulate_bureau_history` from the same
backend namespace. See the [complete example and temporal contracts](docs/temporal.md)
for simulation, date handling, and execution details.
### Workflow reading guide

Start with [Installation](#installation), check your setup with
[Environment Doctor](#environment-doctor), then follow the workflow below.
For time-series inputs, jump to the [complete feature example](#create-features-with-pandas)
and its [feature output summary](#feature-output-summary).

| Step | Guide | What you can do |
| --- | --- | --- |
| 1. Prepare data | [Simulate data](#simulate-data) | Generate sample credit data for experiments. |
| 2. Build features | [Numeric time-series features](#numeric-time-series-features) | Create current values, lags, rolling statistics, and differences from numeric history. |
| 3. Select features | [Feature selection](#feature-selection) | Choose useful inputs with pandas or Spark. |
| 4. Train models | [Model Training](#model-training) | Fit models, configure backends, and save or load fitted models. |
| 5. Validate and tune | [Cross-validation](#cross-validation) and [Optimization](#optimization) | Compare models and search hyperparameters. |
| 6. Evaluate results | [Metrics](#metrics) and [Cube analysis](#cube-analysis) | Evaluate discrimination, risk tables, and performance across segments. |

For implementation details, see [Usage](#usage),
[Spark scalability limits](#spark-scalability-limits),
[Development](#development), [Project layout](#project-layout), and [License](#license).
