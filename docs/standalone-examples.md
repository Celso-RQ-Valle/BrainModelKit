# Standalone examples and detailed workflow alternatives

[Documentation home](README.md) ? [Modeling journeys](../README.md)

These retained examples are **standalone alternatives**, not consecutive cells.
Define the inputs named in each example. A random split called `oot` in an API
example is a holdout, not a chronological out-of-time evaluation. Never use a
final OOT period to select features or tune configurations.

- [Spark simulated data](#spark-simulated-data)
- [Spark features module](#spark-features-module)
- [Spark validation and optimization](#spark-validation-and-optimization)
- [Pandas simulated data](#pandas-simulated-data)
- [Pandas features module](#pandas-features-module)
- [Cross-validation](#cross-validation)
- [Optimization](#optimization)
- [Simulate data](#simulate-data)
- [Feature selection](#feature-selection)
- [Numeric time-series features](#numeric-time-series-features)
- [Workflow reading guide](#workflow-reading-guide)

## Spark simulated data

Generate a synthetic credit table or a per-entity observation history.

Credit simulation accepts `spark_session`, `row_count` (10,000), `feature_count` (20),
`max_days` (30), and `seed` (42), plus `informative_fraction` and `signal_strength`
to control the feature signal. It returns `(credit_df, feature_columns)`.

### Spark credit simulation

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

### Spark history simulation

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

See the [simulation contract](../docs/temporal.md#simulation) for cohort rules and
calendar alignment.

## Spark features module

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

See [feature configuration](../docs/standalone-examples.md#configure-the-generated-features) for nulls,
validation, and legacy API compatibility. Selection then chooses which numeric
columns to send to training; it does not create the temporal features.

### Create features with PySpark

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

### Generate sample data, then create features

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

### Spark feature selection quickstart

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

## Spark validation and optimization

Use the Spark namespaces `brainmodelkit.model_selection.pyspark` and
`brainmodelkit.optimization.pyspark` for CV and Optuna workflows. Their backend
parameters and fold requirements are described in the [CV guide](../docs/model_selection.md)
and [optimization guide](../docs/optimization.md). Keep final OOT evaluation separate
from model selection; read the [shared optimization guidance](../docs/standalone-examples.md#optimization).

## Pandas simulated data

Generate per-entity observation history natively in pandas. The classification
examples in the selection and training sections also show `make_classification`
from scikit-learn for labeled synthetic data; `simulate_credit_data` is Spark-only.

### Pandas history simulation

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

See the [simulation contract](../docs/temporal.md#simulation) for cohort rules and
calendar alignment.

## Pandas features module

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

See [feature configuration](../docs/standalone-examples.md#configure-the-generated-features) for nulls,
validation, and legacy API compatibility. Selection then chooses which numeric
columns to send to training; it does not create the temporal features.

### Create features with Pandas

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

### Feature output summary

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
See [Configure the generated features](../docs/standalone-examples.md#configure-the-generated-features) for
units, composite keys, empty-window behavior, and validation rules.

### Pandas feature selection quickstart

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

## Cross-validation

Use centralized CV on development data; it returns fold metrics and a summary.
Keep OOT data separate from fold creation and model selection.

```python
from brainmodelkit.model_selection.pandas import cross_validate

result = cross_validate(train_df, "target", features, "logistic_regression", cv=5)
print(result.fold_metrics)
print(result.summary)
```

See the [cross-validation guide](../docs/model_selection.md) for K-Fold,
stratification, groups, time ordering, Spark folds, and leakage guidance.

## Optimization

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
temporal holdout for final evaluation.** See the [optimization guide](../docs/optimization.md)
for metric names, selection semantics, diagnostics, and Spark backend setup.

## Simulate data

Choose [Spark simulation](../docs/standalone-examples.md#spark-simulated-data) or [Pandas simulation](../docs/standalone-examples.md#pandas-simulated-data).

## Feature selection

Start here to choose features, inspect the reasons, and evaluate a final model.
All sixteen methods are available in both Pandas and Spark. Backend algorithms
and arguments can differ; each method link below explains the actual computation,
assumptions, decision rules, diagnostics, and examples.

Examples: [Spark feature selection quickstart](../docs/standalone-examples.md#spark-feature-selection-quickstart) and [Pandas feature selection quickstart](../docs/standalone-examples.md#pandas-feature-selection-quickstart).

### Choose a method and read its algorithm

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

## Numeric time-series features

Use `build_features(df_base, df_spine, ...)` to create current
values, lags, and rolling statistics from historical numeric data. Pass the
historical source **first** and the rows to enrich (the spine) **second**:

- `df_base` contains entity keys, observation dates, and numeric values.
- `df_spine` contains entity keys and a reference date for each row to enrich.
  Its original rows and columns are preserved.

### Configure the generated features

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
backend namespace. See the [complete example and temporal contracts](../docs/temporal.md)
for simulation, date handling, and execution details.
## Workflow reading guide

Start with [Installation](../docs/setup.md#installation), check your setup with
[Environment Doctor](../docs/setup.md#environment-doctor), then follow the workflow below.
For time-series inputs, jump to the [complete feature example](../docs/standalone-examples.md#create-features-with-pandas)
and its [feature output summary](../docs/standalone-examples.md#feature-output-summary).

| Step | Guide | What you can do |
| --- | --- | --- |
| 1. Prepare data | [Simulate data](../docs/standalone-examples.md#simulate-data) | Generate sample credit data for experiments. |
| 2. Build features | [Numeric time-series features](../docs/standalone-examples.md#numeric-time-series-features) | Create current values, lags, rolling statistics, and differences from numeric history. |
| 3. Select features | [Feature selection](../docs/standalone-examples.md#feature-selection) | Choose useful inputs with pandas or Spark. |
| 4. Train models | [Model Training](../docs/training.md#model-training) | Fit models, configure backends, and save or load fitted models. |
| 5. Validate and tune | [Cross-validation](../docs/standalone-examples.md#cross-validation) and [Optimization](../docs/standalone-examples.md#optimization) | Compare models and search hyperparameters. |
| 6. Evaluate results | [Metrics](../docs/evaluation.md#metrics) and [Cube analysis](../docs/evaluation.md#cube-analysis) | Evaluate discrimination, risk tables, and performance across segments. |

For implementation details, see [Usage](../docs/setup.md#usage),
[Spark scalability limits](../docs/setup.md#spark-scalability-limits),
[Development](../docs/project.md#development), [Project layout](../docs/project.md#project-layout), and [License](../docs/project.md#license).
