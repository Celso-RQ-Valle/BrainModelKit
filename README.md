# BrainModelKit

BrainModelKit builds numeric time-series features, selects model inputs, trains
binary classifiers, and evaluates overall and segment performance. Choose the
backend that matches your DataFrames; the base install has no runtime dependencies.

> [!WARNING]
> The public API is not stable yet. BrainModelKit is under active development
> and remains experimental, including its Spark integrations.

| Choose | Data and execution | Start here |
| --- | --- | --- |
| Spark | Spark DataFrames and native distributed operations; individual algorithms have driver/worker limits. | [Spark guide](#spark-guide) |
| pandas | In-memory pandas DataFrames and scikit-learn-compatible models. | [pandas guide](#pandas-guide) |

The journey is **install ? diagnose ? prepare ? build ? select ? train ?
validate/tune ? evaluate ? analyze segments ? save/load/score**. Each chapter
follows this order. Its snippets continue one workflow unless marked as an
alternative. [Complete runnable examples](examples/journey/README.md) include
imports, missing-value handling, and assertions using a separate small fixture.
The README workflow below uses the project's credit and bureau-history simulators,
also demonstrated in [the credit notebook](notebooks/simulate_credit_data.ipynb).

## Contents

- [Spark guide](#spark-guide)
  - [Install and diagnose](#spark-install-and-diagnose)
  - [Prepare data](#spark-prepare-data)
  - [Build and select features](#spark-features)
  - [Train a baseline](#spark-train-a-baseline)
  - [Validate and optimize](#spark-validate-and-optimize)
  - [Evaluate metrics](#spark-evaluate-metrics)
  - [Analyze segments](#spark-analyze-segments)
  - [Save, load, and score](#spark-save-load-and-score)
- [pandas guide](#pandas-guide)
  - [Install and diagnose](#pandas-install-and-diagnose)
  - [Prepare data](#pandas-prepare-data)
  - [Build and select features](#pandas-features)
  - [Train a baseline](#pandas-train-a-baseline)
  - [Validate and optimize](#pandas-validate-and-optimize)
  - [Evaluate metrics](#pandas-evaluate-metrics)
  - [Analyze segments](#pandas-analyze-segments)
  - [Save, load, and score](#pandas-save-load-and-score)
- [Reference and troubleshooting](#reference-and-troubleshooting)
- [Project information](#project-information)

## Spark guide

### Spark install and diagnose

```bash
python -m pip install "brainmodelkit[pyspark]"
```

Python **3.10+** is the packaging minimum. Spark also needs compatible Java and a configured Spark runtime (PySpark >=3.5); consult the tested matrix rather than treating the packaging minimum as universal compatibility.
For unreleased checkout examples, use `python -m pip install -e ".[pyspark]"`.

```python
import brainmodelkit as bmk

report = bmk.doctor(
    backend="pyspark", model="logistic_regression", save_model_to="none"
)
```

`READY` means inspected prerequisites are present; it does not certify an actual
fit, native library execution, credentials, or artifact writes. `PARTIAL` identifies
checks that could not be completed. See [Doctor and setup](docs/setup.md#environment-doctor)
and the [tested compatibility matrix](docs/release-readiness.md).
Keep managed Spark versions; see [managed installs and SynapseML setup](docs/setup.md#installation). LightGBM execution choices are explained under training below.

### Spark prepare data

**Continuing the main workflow.** Use the project's `simulate_credit_data` and
`simulate_bureau_history` APIs. The credit simulator supplies the binary default
label, company-size segments, and 20 numeric inputs. Its noisy class-conditional
signal is for demonstrations, not a production labeling recipe or benchmark.
Bureau histories are simulated independently of the credit label.

```python
from datetime import date

from pyspark.sql import SparkSession, functions as F
from brainmodelkit.pyspark import simulate_credit_data, simulate_bureau_history

spark = SparkSession.builder.appName("BrainModelKitJourney").getOrCreate()
credit, credit_features = simulate_credit_data(
    spark, row_count=1200, feature_count=20, max_days=90, seed=42
)
# The simulator dates are relative to today. Rebase them for fixed tutorial splits.
first_date = credit.agg(F.min("reference_date")).first()[0]
df_spine = (
    credit.withColumn(
        "reference_date",
        F.date_add(F.lit(date(2026, 4, 1)),
                   F.datediff("reference_date", F.lit(first_date))),
    )
    .withColumnRenamed("company_id", "account_id")
    .withColumnRenamed("default_flag", "target")
    .withColumnRenamed("company_size", "segment")
)
df_base = simulate_bureau_history(
    df_spine.select("account_id"), entity_cols="account_id",
    initial_date="2025-10-01", final_date="2026-06-30", seed=42,
)

# Optional: export this small credit sample for the pandas guide (requires pandas).
df_spine.toPandas().to_csv("simulated_credit.csv", index=False)
```

| Frame | Required here | Role |
| --- | --- | --- |
| `df_base` | `account_id`, `observation_date`, numeric `bureau_score` | Simulated historical source; no duplicate entity/date pairs. |
| `df_spine` | `account_id`, `reference_date`, binary `target`, `segment`, `feature_01` through `feature_20` | Simulated credit records; target 1 is default. Segment is company size and is used only for diagnostics. |

April is development training, May is validation, and June is final OOT.
These are chronological partitions of simulated dates, not evidence of real
out-of-time performance. With real data, use labels whose outcomes are observable
for each training period and inputs available by the reference date.
See the [credit simulation](docs/standalone-examples.md#spark-credit-simulation)
and [history simulation](docs/standalone-examples.md#spark-history-simulation) options.

### Spark features

#### Spark build features

Building creates explanatory variables; selection later chooses which enter a
model. Every reference row uses matching entity keys and its own reference date.

```python
from brainmodelkit.pyspark import build_features

enriched = build_features(
    df_base,
    df_spine,
    key_cols="account_id",
    value_cols=["bureau_score"],
    reference_date_col="reference_date",
    windows=[1, 3, 6],
    lags=[1, 2, 3],
    period_unit="months",
    aggregations=["mean", "median", "min", "max", "std", "sum", "count"],
)
history_features = [c for c in enriched.columns if c not in df_spine.columns]
candidate_features = credit_features + history_features
```

`date_col` defaults to `observation_date`. Windows are **(reference ? k periods,
reference]**; lags use the latest non-null value at or before reference ? k periods.
Units are days, seven-day weeks, or calendar months (clipped at month end).
Supported aggregations are mean, exact median, min, max, sample std, **sum**, and
count. Each source gets current/lag values, `_date_diff_days` latency relative
to the spine reference date, and `current_minus_mean` for every window.
No generated lookup dates are returned. The spine's columns and duplicate rows remain.

This configuration generates **32 bureau features** plus the 20 credit inputs.
History starts in October 2025 so the April cohort can use the six-month window.
All seven supported aggregates are requested explicitly (they are also the API defaults).

| Generated family | Count | Example column |
| --- | ---: | --- |
| Current value and observation age | 2 | `bureau_score__current`, `bureau_score__current_date_diff_days` |
| Three lag values and their observation ages | 6 | `bureau_score__lag_2_months`, `bureau_score__lag_2_months_date_diff_days` |
| Seven aggregates across three windows | 21 | `bureau_score__median_3_months`, `bureau_score__std_6_months` |
| Current minus each window mean | 3 | `bureau_score__current_minus_mean_3_months` |

`std` is sample standard deviation (`ddof=1`); fewer than two observations
produce null. Median is exact. Observation ages are measured from the spine
reference date, including for lagged values.


**Optional relationship variant.** Run this immediately after the build above,
before filling missing values or splitting the data. It adds eight numeric
relationships to the 52 existing candidates. `build_features` already computes
current-minus-mean; the following are explicit calculations on its outputs.
They use only history available at each row's reference date.

| Added relationship | Calculation |
| --- | --- |
| Change from one-month lag | Current minus lag 1 |
| Ratio to one-month lag | Current / lag 1 |
| Relative change from one-month lag | (current minus lag 1) / lag 1; multiply by 100 for percent |
| Difference from median | Current minus three-month median |
| Three-month range | Three-month maximum minus minimum |
| Standardized deviation | (current minus three-month mean) / three-month sample std |
| Coefficient of variation | Three-month sample std / absolute three-month mean |
| Short/long mean ratio | One-month mean / six-month mean |

Missing inputs and zero denominators remain null until the workflow's explicit
imputation step. A zero-variance window has an undefined standardized deviation.
These ratios compare a source with its own history; choose business-meaningful
pairs when extending the recipe to different source columns.

```python
from pyspark.sql import functions as F

current = enriched["bureau_score__current"]
lag = enriched["bureau_score__lag_1_months"]
mean = enriched["bureau_score__mean_3_months"]
std = enriched["bureau_score__std_3_months"]


def safe_ratio(numerator, denominator):
    return numerator / F.when(denominator != 0, denominator)


relationships = {
    "bureau_score__change_1_month": current - lag,
    "bureau_score__ratio_1_month": safe_ratio(current, lag),
    "bureau_score__relative_change_1_month": safe_ratio(current - lag, lag),
    "bureau_score__current_minus_median_3_months": (
        current - enriched["bureau_score__median_3_months"]
    ),
    "bureau_score__range_3_months": (
        enriched["bureau_score__max_3_months"] - enriched["bureau_score__min_3_months"]
    ),
    "bureau_score__standardized_deviation_3_months": safe_ratio(current - mean, std),
    "bureau_score__coefficient_of_variation_3_months": safe_ratio(std, F.abs(mean)),
    "bureau_score__mean_ratio_1_to_6_months": safe_ratio(
        enriched["bureau_score__mean_1_months"], enriched["bureau_score__mean_6_months"]
    ),
}
enriched = enriched.select("*", *[expr.alias(name) for name, expr in relationships.items()])
candidate_features = credit_features + history_features + list(relationships)
enriched.select("account_id", "reference_date", *relationships).show(5)
```

Missing lookups/statistics are null; empty windows have count zero, including a
null sum. This recipe uses fixed zero filling as an explicit demonstration policy;
it learns no statistics from validation or OOT. In production, fit any learned
imputer/encoder on training data only and inside folds for CV. Keys, dates, target,
segment, status, and industry section never enter `candidate_features`.
The inputs combine the simulator's numeric credit columns with generated bureau features.

```python
enriched = enriched.fillna(0, subset=candidate_features)
train_df = enriched.filter("reference_date < '2026-05-01'")
validation_df = enriched.filter(
    "reference_date >= '2026-05-01' AND reference_date < '2026-06-01'"
)
oot_df = enriched.filter("reference_date >= '2026-06-01'")
```

Spark joins can expand sharply with many reference dates per entity; unlimited-staleness lookups retain all earlier history. Validation adds scans/shuffles; exact median needs group memory.
The [full feature catalog](docs/temporal.md#temporal-semantics) covers naming,
all aggregations, staleness, boundaries, and validation. There is no `features="all"`
option, fitted trend family, or automatic cross-column relationship family in this API.
The relationship variant above demonstrates how to extend its outputs explicitly.

#### Spark select features

Fit selectors on development training rows only, then carry their ordered output
into training. This minimal filter removes constants; it is not supervised ranking.

```python
from brainmodelkit.feature_selection.pyspark import variance_filter

selection = variance_filter(train_df, feature_cols=candidate_features)
selected_features = selection.selected_features
if not selected_features:
    raise ValueError("No features survived; inspect selection.feature_table")
```

Start with unusable inputs, examine redundancy/univariate signal when appropriate,
then consider model-based and expensive iterative methods. Do not run every method
by default. Tiers below are **approximate and overlapping**, derived from scans,
sorts, fits, folds, and resampling in the implementations, not benchmark timings.

| Method | Purpose | Relative cost | Main cost driver | When to use | Guide |
| --- | --- | --- | --- | --- | --- |
| `completeness` | Availability | Low | Non-null aggregates; grouped shuffles | Missing-input screening | [Details](docs/feature_selection/completeness.md) |
| `variance_filter` | Remove constants | Low | Native var_pop aggregates | Before fitting | [Details](docs/feature_selection/variance_filter.md) |
| `cardinality` | Distinct values | Low?moderate | Exact distinct counts and shuffles | Before encoding | [Details](docs/feature_selection/cardinality.md) |
| `anova` | Class-mean signal | Low?moderate | Two class summaries per feature | Numeric screening | [Details](docs/feature_selection/anova.md) |
| `chi_square` | Count/category association | Low?moderate | Categorical contingency tables | Counts (pandas); category codes (Spark) | [Details](docs/feature_selection/chi_square.md) |
| `information_value` | Binary distribution separation | Moderate | Quantiles and grouped bin counts | Inspect WoE bins | [Details](docs/feature_selection/information_value.md) |
| `mutual_information` | Nonlinear univariate signal | Moderate | Quantile binning and joint-count shuffles | Nonlinear screening | [Details](docs/feature_selection/mutual_information.md) |
| `correlation_filter` | Reduce redundancy | Moderate?high | Distributed matrix/ranks; p? driver memory | After quality checks | [Details](docs/feature_selection/correlation_filter.md) |
| `l1_selection` | Sparse linear subset | Moderate?high | Distributed logistic iterations | Compact linear models | [Details](docs/feature_selection/l1_selection.md) |
| `feature_importance_selection` | Model-based ranking | Moderate?high | One native model fit | Capture model-specific signal | [Details](docs/feature_selection/feature_importance_selection.md) |
| `permutation_importance_selection` | Held-out reliance | High | Repeats ? sorting, joins, predictions | Validate model reliance | [Details](docs/feature_selection/permutation_importance_selection.md) |
| `rfe` | Recursive elimination | High | Distributed fits per step; reports | Conditional importance | [Details](docs/feature_selection/rfe.md) |
| `stability_selection` | Selection repeatability | High?very high | Poisson resample fits ? groups | Check robustness | [Details](docs/feature_selection/stability_selection.md) |
| `rfecv` | Choose subset size | Very high | Folds ? elimination fits plus refit | Budgeted subset search | [Details](docs/feature_selection/rfecv.md) |
| `sequential_selection` | Greedy subset search | Very high | Candidate subsets ? distributed fits | Small candidate pools | [Details](docs/feature_selection/sequential_selection.md) |
| `boruta` | All-relevant selection | Very high | Shadow sorts/joins + repeated forests | All-relevant hypothesis testing | [Details](docs/feature_selection/boruta.md) |

Spark collects bounded statistics and model summaries, including a feature-pair correlation matrix and bounded stability group keys. Permutation and Boruta require classic Spark Python workers, not Spark Connect; distribution is not a guarantee of lower cost.
See [cost evidence and backend differences](docs/selection-costs.md) and
[selection contracts](docs/feature_selection.md) for target requirements and diagnostics.

### Spark train a baseline

Pass raw selected numeric columns. The target must be 0/1 with both classes in
training. `model_params` contains estimator settings; predictions add `score`,
the class-1 probability. Here the API argument `oot_df` receives **validation_df**:
its `oot_*` metric names do not make this development set final OOT.

```python
from brainmodelkit.training.pyspark import train_model

baseline_params = {"regParam": 0.01, "maxIter": 10}
result = train_model(
    "baseline",
    "target",
    selected_features,
    train_df,
    validation_df,
    model="logistic_regression",
    model_params=baseline_params,
    save_model_to="none",
)
print(result.metrics)
```

`TrainingResult` contains the fitted model, metrics, feature importance,
`oot_predictions`, optional `scoring_predictions`, and persistence metadata.
Inspect the baseline before adding complexity; use the [full training contract](docs/training.md#training-parameter-reference)
for all parameters and [output fields](docs/training.md#training-outputs).

| Model | Public identifier | Dependency/backend | Key notes | Example settings in `model_params` |
| --- | --- | --- | --- | --- |
| Logistic regression | `logistic_regression` | Spark ML | Linear probability classifier | `{"regParam": 0.1, "maxIter": 20}` |
| Decision tree | `decision_tree` | Spark ML | Single tree | `{"maxDepth": 4}` |
| Random forest | `random_forest` | Spark ML | Tree ensemble | `{"numTrees": 20, "maxDepth": 5}` |
| Gradient boosting | `gradient_boosting` | Spark ML GBT | Binary boosted trees | `{"maxIter": 20}` |
| LightGBM | `lightgbm` / `lgbm` | SparkLightGBM or SynapseML | Backend choice below | `{"num_iterations": 20}` for default backend |
| Custom estimator | Estimator object | Compatible Spark ML stage | Must expose required probability/feature/label parameters | Native `setParams` names |

Pass `lgbm_backend="default"` or `"synapseml"` directly to `train_model`,
`cross_validate`, or `optimize`, **not** inside `model_params`. Default SparkLightGBM
uses native names (`num_iterations`, `learning_rate`); SynapseML uses camelCase
(`numIterations`, `learningRate`) and matching JVM artifacts.
Default local SparkLightGBM can collect training rows into driver NumPy arrays;
`num_workers>1` enables distributed native training with executor/barrier constraints.
See [LightGBM training examples](docs/training.md#choosing-the-lightgbm-backend)
and [runtime requirements](docs/optimization.md#spark-lightgbm).

### Spark validate and optimize

**Continuing the main workflow.** Compare the baseline with one alternative on
May validation; do not look at June OOT while choosing. This small manual search
preserves the chronological split and the training-only selected subset.

```python
alternative_params = {"regParam": 0.1, "maxIter": 10}
alternative = train_model(
    "alternative",
    "target",
    selected_features,
    train_df,
    validation_df,
    model="logistic_regression",
    model_params=alternative_params,
    save_model_to="none",
)
best_params = (
    alternative_params
    if alternative.metrics["oot_auc"] > result.metrics["oot_auc"]
    else baseline_params
)
development_df = train_df.unionByName(validation_df)
```

Freeze the feature list and chosen settings. The next step refits on development
training + validation and evaluates June once. CV instead estimates performance
across development folds; Optuna optimization evaluates multiple configurations.
See [verified CV and optimization alternatives](docs/journey-validation.md#spark-api-alternatives)
for `cross_validate` / `optimize`, search spaces, and final refitting.
These alternatives use fixed candidate features, not selection fitted outside CV.
Spark time-ordered CV still trains on all other folds, including future ones; it is not expanding-window validation. Use explicit forward splits for temporal evaluation.
The optimizer accepts no fold/preprocessing callbacks; its default CV is stratified.
Never pass the final OOT set as optimization's `oot_df`; pass development validation.

### Spark evaluate metrics

**Continuing:** this is the first model decision-independent use of June OOT.

```python
result = train_model(
    "final",
    "target",
    selected_features,
    development_df,
    oot_df,
    model="logistic_regression",
    model_params=best_params,
    save_model_to="none",
)
predictions = result.oot_predictions
print(result.metrics)
```

Targets are binary 0/1. AUC/Gini expect higher scores for class 1; ranking scores
need not be calibrated probabilities. The model's `score` is a probability.
KS measures separation magnitude, not calibration or direction. Scalars are in
decimal units; Gini is `2 * AUC - 1`. Undefined metrics are NaN for single-class
or empty valid pairs (Spark grouped output may have no row for absent groups).

| Metric or utility | What it answers | Interpretation | Usage/example |
| --- | --- | --- | --- |
| `ks_ntile` | How separated are class distributions? | 0?1; larger means more separation; tile approximation. | [Spark KS](docs/evaluation.md#spark-kolmogorov-smirnov-ks) |
| `auc_gini` | How well are positives ranked? | AUC 0.5 is chance ranking, 1 is perfect; Gini 0 and 1 respectively. | [AUC/Gini](docs/evaluation.md#spark-roc-auc-and-gini) |
| `roc_auc_gini` | Same AUC/Gini calculation | Public alias of `auc_gini`, same signature/shape. | [Alias example](docs/evaluation.md#public-alias-and-scalar-utility) |
| `curve_roc` | Tradeoff across thresholds | DataFrame of curve points; plot separately. | [ROC example](docs/evaluation.md#spark-plot-a-roc-curve) |
| `risk_table` | Event rate across ordered score tiles | DataFrame of occupied tiles, ranges, volumes, and event rates. | [Risk table](docs/evaluation.md#spark-risk-sorting-table) |

```python
from brainmodelkit.metrics.pyspark import auc_gini, ks_ntile

ks_ntile(dataframe=predictions, group_columns="segment", n_tiles=10).show()
auc_gini(df=predictions, n_tiles=10).show()
```

Spark's KS/AUC/ROC are tile approximations; `n_tiles`, ties, and partition ordering can change results. Grouped `auc_gini` collects distinct group keys and evaluates groups separately. For many groups prefer the native cube API below. Only ROC points need collecting for plotting.
See [metric input and undefined-case details](docs/evaluation.md).

### Spark analyze segments

Overall performance can hide weaker segments. Start with a few meaningful,
adequately populated dimensions from `predictions`.

```python
from brainmodelkit.cube_analysis.pyspark import calculate_metrics, calculate_ntile

cube = calculate_metrics(df=predictions, group_columns=["segment"])
tile_cube = calculate_ntile(df=predictions, group_columns=["segment"], n_tiles=4)
cube.show()
tile_cube.show()
```

Metric rows contain `segment`, `KS`, `AUC`, `Gini`, `score`; `score` here is the
**source score column name**. `Geral` denotes the overall rollup. With more
dimensions, omitted dimensions also contain `Geral`; actual missing keys remain
missing. Each score requires `2 ** dimensions` grouping calculations.
Risk cubes additionally report volumes/events per occupied tile; use these to
judge support. Small groups are noisy, single-class metrics are undefined, and
there is no automatic minimum-volume suppression. Real keys named `Geral` collide
visually with rollups. [Cube interpretation](docs/evaluation.md#interpreting-segment-results)
includes an illustrative result and full output conventions.

### Spark save, load, and score

The walkthrough used `save_model_to="none"`. To persist the frozen final recipe,
run that final fit with folder persistence enabled (or use `--save-dir` in the
[separate fixture-based runnable example](examples/journey/README.md)). This does not choose new settings.

```python
saved_result = train_model(
    "saved_final",
    "target",
    selected_features,
    development_df,
    oot_df,
    model="logistic_regression",
    model_params=best_params,
    save_model_to="folder",
    model_format="spark",
    output_dir="training_runs",
)
print(saved_result.output_dir, saved_result.model_uri)

from brainmodelkit.persistence import load_model
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

loaded = load_model(saved_result.model_uri, format="spark", model_class=PipelineModel)
new_data = oot_df.drop("target")  # Schema demonstration; replace with future rows.
scored = loaded.transform(new_data).withColumn(
    "score", vector_to_array("probability")[1]
)
scored.select("account_id", "score").show()
```

Folder mode writes the model plus metadata, metrics, and feature-importance reports
in a unique run directory. Preserve the feature recipe, preprocessing policy, types,
and order; the persisted estimator does not reconstruct history or build features.
Spark pipelines contain feature assembly and accept raw named columns. Folder writes require a working Hadoop filesystem; see [Windows setup](docs/setup.md#spark-on-windows).
MLflow uses its own model flavors; `none` writes no artifacts. Native LightGBM,
ONNX, and Spark loaders do not return the same object type or prediction interface.
See [persistence formats and loaders](docs/training.md#model-formats),
[MLflow](docs/training.md#mlflow), and [legacy migration](docs/training.md#migration-and-compatibility).

## pandas guide

### pandas install and diagnose

```bash
python -m pip install "brainmodelkit[pandas]"
```

Python **3.10+** is the packaging minimum. The pandas extra includes pandas >=2.0, NumPy, scikit-learn, and native LightGBM.
For unreleased checkout examples, use `python -m pip install -e ".[pandas]"`.

```python
import brainmodelkit as bmk

report = bmk.doctor(backend="pandas", model="logistic_regression", save_model_to="none")
```

`READY` means inspected prerequisites are present; it does not certify an actual
fit, native library execution, credentials, or artifact writes. `PARTIAL` identifies
checks that could not be completed. See [Doctor and setup](docs/setup.md#environment-doctor)
and the [tested compatibility matrix](docs/release-readiness.md).
Add optional extras only for the capabilities you use, such as `optimization`, `mlflow`, `persistence`, or `boruta`; see [installation options](docs/setup.md#installation).

### pandas prepare data

**Continuing the main workflow.** Load `simulated_credit.csv` exported in
[Spark prepare data](#spark-prepare-data), then generate bureau histories with the
pandas simulator. `simulate_credit_data` is currently Spark-only: generating the
CSV requires the Spark setup above; running this guide with an existing export
requires only the pandas backend. Both guides use the same credit rows, labels,
segments, numeric inputs, and April/May/June date splits.

```python
import pandas as pd
from brainmodelkit.pandas import simulate_bureau_history

df_spine = pd.read_csv(
    "simulated_credit.csv", dtype={"account_id": "string"},
    parse_dates=["reference_date"],
)
credit_features = [f"feature_{i:02d}" for i in range(1, 21)]
df_base = simulate_bureau_history(
    df_spine[["account_id"]], entity_cols="account_id",
    initial_date="2025-10-01", final_date="2026-06-30", seed=42,
)
```

| Frame | Required here | Role |
| --- | --- | --- |
| `df_base` | `account_id`, `observation_date`, numeric `bureau_score` | Simulated historical source; no duplicate entity/date pairs. |
| `df_spine` | `account_id`, `reference_date`, binary `target`, `segment`, `feature_01` through `feature_20` | Simulated credit records; target 1 is default. Segment is company size and is used only for diagnostics. |

April is development training, May is validation, and June is final OOT.
The credit inputs have simulated class-conditional signal; bureau histories are
independent of the label. These chronological partitions are a workflow demo,
not a performance benchmark. With real data, use labels observable for each
training period and values available by the reference date.
See [history simulation options](docs/standalone-examples.md#pandas-history-simulation).

### pandas features

#### pandas build features

Building creates explanatory variables; selection later chooses which enter a
model. Every reference row uses matching entity keys and its own reference date.

```python
from brainmodelkit.pandas import build_features

enriched = build_features(
    df_base,
    df_spine,
    key_cols="account_id",
    value_cols=["bureau_score"],
    reference_date_col="reference_date",
    windows=[1, 3, 6],
    lags=[1, 2, 3],
    period_unit="months",
    aggregations=["mean", "median", "min", "max", "std", "sum", "count"],
)
history_features = [c for c in enriched.columns if c not in df_spine.columns]
candidate_features = credit_features + history_features
```

`date_col` defaults to `observation_date`. Windows are **(reference ? k periods,
reference]**; lags use the latest non-null value at or before reference ? k periods.
Units are days, seven-day weeks, or calendar months (clipped at month end).
Supported aggregations are mean, exact median, min, max, sample std, **sum**, and
count. Each source gets current/lag values, `_date_diff_days` latency relative
to the spine reference date, and `current_minus_mean` for every window.
No generated lookup dates are returned. The spine's columns and duplicate rows remain.

This configuration generates **32 bureau features** plus the 20 credit inputs.
History starts in October 2025 so the April cohort can use the six-month window.
All seven supported aggregates are requested explicitly (they are also the API defaults).

| Generated family | Count | Example column |
| --- | ---: | --- |
| Current value and observation age | 2 | `bureau_score__current`, `bureau_score__current_date_diff_days` |
| Three lag values and their observation ages | 6 | `bureau_score__lag_2_months`, `bureau_score__lag_2_months_date_diff_days` |
| Seven aggregates across three windows | 21 | `bureau_score__median_3_months`, `bureau_score__std_6_months` |
| Current minus each window mean | 3 | `bureau_score__current_minus_mean_3_months` |

`std` is sample standard deviation (`ddof=1`); fewer than two observations
produce null. Median is exact. Observation ages are measured from the spine
reference date, including for lagged values.


**Optional relationship variant.** Run this immediately after the build above,
before filling missing values or splitting the data. It adds eight numeric
relationships to the 52 existing candidates. `build_features` already computes
current-minus-mean; the following are explicit calculations on its outputs.
They use only history available at each row's reference date.

| Added relationship | Calculation |
| --- | --- |
| Change from one-month lag | Current minus lag 1 |
| Ratio to one-month lag | Current / lag 1 |
| Relative change from one-month lag | (current minus lag 1) / lag 1; multiply by 100 for percent |
| Difference from median | Current minus three-month median |
| Three-month range | Three-month maximum minus minimum |
| Standardized deviation | (current minus three-month mean) / three-month sample std |
| Coefficient of variation | Three-month sample std / absolute three-month mean |
| Short/long mean ratio | One-month mean / six-month mean |

Missing inputs and zero denominators remain null until the workflow's explicit
imputation step. A zero-variance window has an undefined standardized deviation.
These ratios compare a source with its own history; choose business-meaningful
pairs when extending the recipe to different source columns.

```python
current = enriched["bureau_score__current"]
lag = enriched["bureau_score__lag_1_months"]
mean = enriched["bureau_score__mean_3_months"]
std = enriched["bureau_score__std_3_months"]


def safe_ratio(numerator, denominator):
    return numerator / denominator.where(denominator.ne(0))


relationships = {
    "bureau_score__change_1_month": current - lag,
    "bureau_score__ratio_1_month": safe_ratio(current, lag),
    "bureau_score__relative_change_1_month": safe_ratio(current - lag, lag),
    "bureau_score__current_minus_median_3_months": (
        current - enriched["bureau_score__median_3_months"]
    ),
    "bureau_score__range_3_months": (
        enriched["bureau_score__max_3_months"] - enriched["bureau_score__min_3_months"]
    ),
    "bureau_score__standardized_deviation_3_months": safe_ratio(current - mean, std),
    "bureau_score__coefficient_of_variation_3_months": safe_ratio(std, mean.abs()),
    "bureau_score__mean_ratio_1_to_6_months": safe_ratio(
        enriched["bureau_score__mean_1_months"], enriched["bureau_score__mean_6_months"]
    ),
}
enriched = enriched.assign(**relationships)
candidate_features = credit_features + history_features + list(relationships)
print(enriched[["account_id", "reference_date", *relationships]].head())
```

Missing lookups/statistics are null; empty windows have count zero, including a
null sum. This recipe uses fixed zero filling as an explicit demonstration policy;
it learns no statistics from validation or OOT. In production, fit any learned
imputer/encoder on training data only and inside folds for CV. Keys, dates, target,
segment, status, and industry section never enter `candidate_features`.
The inputs combine the simulator's numeric credit columns with generated bureau features.

```python
enriched[candidate_features] = enriched[candidate_features].fillna(0)
ref = pd.to_datetime(enriched.reference_date)
train_df = enriched.loc[ref < "2026-05-01"].copy()
validation_df = enriched.loc[(ref >= "2026-05-01") & (ref < "2026-06-01")].copy()
oot_df = enriched.loc[ref >= "2026-06-01"].copy()
```

Pandas sorts history once, uses binary-search boundaries, and caches distinct entity/reference results. Window reductions and memory still grow with requested features and history.
The [full feature catalog](docs/temporal.md#temporal-semantics) covers naming,
all aggregations, staleness, boundaries, and validation. There is no `features="all"`
option, fitted trend family, or automatic cross-column relationship family in this API.
The relationship variant above demonstrates how to extend its outputs explicitly.

Run the [complete relationship example](examples/temporal/feature_relationships.py)
against the simulated credit export:

```bash
python examples/temporal/feature_relationships.py simulated_credit.csv
```

#### pandas select features

Fit selectors on development training rows only, then carry their ordered output
into training. This minimal filter removes constants; it is not supervised ranking.

```python
from brainmodelkit.feature_selection.pandas import variance_filter

selection = variance_filter(train_df, feature_cols=candidate_features)
selected_features = selection.selected_features
if not selected_features:
    raise ValueError("No features survived; inspect selection.feature_table")
```

Start with unusable inputs, examine redundancy/univariate signal when appropriate,
then consider model-based and expensive iterative methods. Do not run every method
by default. Tiers below are **approximate and overlapping**, derived from scans,
sorts, fits, folds, and resampling in the implementations, not benchmark timings.

| Method | Purpose | Relative cost | Main cost driver | When to use | Guide |
| --- | --- | --- | --- | --- | --- |
| `completeness` | Availability | Low | Non-null scans; optional grouped aggregates | Missing-input screening | [Details](docs/feature_selection/completeness.md) |
| `variance_filter` | Remove constants | Low | Column moments | Before fitting | [Details](docs/feature_selection/variance_filter.md) |
| `cardinality` | Distinct values | Low?moderate | Exact per-column distinct sets | Before encoding | [Details](docs/feature_selection/cardinality.md) |
| `anova` | Class-mean signal | Low?moderate | Class-wise sums and variances | Numeric screening | [Details](docs/feature_selection/anova.md) |
| `chi_square` | Count/category association | Low?moderate | Count-by-class summaries | Counts (pandas); category codes (Spark) | [Details](docs/feature_selection/chi_square.md) |
| `information_value` | Binary distribution separation | Moderate | Binning and class counts | Inspect WoE bins | [Details](docs/feature_selection/information_value.md) |
| `mutual_information` | Nonlinear univariate signal | Moderate | Nearest-neighbor estimation for continuous data | Nonlinear screening | [Details](docs/feature_selection/mutual_information.md) |
| `correlation_filter` | Reduce redundancy | Moderate?high | Feature-pair matrix; Spearman ranks | After quality checks | [Details](docs/feature_selection/correlation_filter.md) |
| `l1_selection` | Sparse linear subset | Moderate?high | Iterative penalized logistic fit | Compact linear models | [Details](docs/feature_selection/l1_selection.md) |
| `feature_importance_selection` | Model-based ranking | Moderate?high | One estimator fit | Capture model-specific signal | [Details](docs/feature_selection/feature_importance_selection.md) |
| `permutation_importance_selection` | Held-out reliance | High | Features ? repeats ? predictions | Validate model reliance | [Details](docs/feature_selection/permutation_importance_selection.md) |
| `rfe` | Recursive elimination | High | Fits per elimination step; reports | Conditional importance | [Details](docs/feature_selection/rfe.md) |
| `stability_selection` | Selection repeatability | High?very high | Bootstrap fits ? groups | Check robustness | [Details](docs/feature_selection/stability_selection.md) |
| `rfecv` | Choose subset size | Very high | Folds ? elimination fits | Budgeted subset search | [Details](docs/feature_selection/rfecv.md) |
| `sequential_selection` | Greedy subset search | Very high | Candidate subsets ? folds | Small candidate pools | [Details](docs/feature_selection/sequential_selection.md) |
| `boruta` | All-relevant selection | Very high | Forests on real + shadow columns | All-relevant hypothesis testing | [Details](docs/feature_selection/boruta.md) |

Pandas works in memory. MI uses sklearn estimation; chi-square expects integer counts/indicators; L1 uses `C`. These differ from Spark binning, categorical tests, and `reg_param`.
See [cost evidence and backend differences](docs/selection-costs.md) and
[selection contracts](docs/feature_selection.md) for target requirements and diagnostics.

### pandas train a baseline

Pass raw selected numeric columns. The target must be 0/1 with both classes in
training. `model_params` contains estimator settings; predictions add `score`,
the class-1 probability. Here the API argument `oot_df` receives **validation_df**:
its `oot_*` metric names do not make this development set final OOT.

```python
from brainmodelkit.training.pandas import train_model
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Fit scaling on training rows only; credit inputs and bureau scores differ in scale.
classifier = make_pipeline(StandardScaler(), LogisticRegression())

baseline_params = {"logisticregression__C": 0.1, "logisticregression__max_iter": 1000}
result = train_model(
    "baseline",
    "target",
    selected_features,
    train_df,
    validation_df,
    model=classifier,
    model_params=baseline_params,
    save_model_to="none",
)
print(result.metrics)
```

`TrainingResult` contains the fitted model, metrics, feature importance,
`oot_predictions`, optional `scoring_predictions`, and persistence metadata.
Inspect the baseline before adding complexity; use the [full training contract](docs/training.md#training-parameter-reference)
for all parameters and [output fields](docs/training.md#training-outputs).

| Model | Public identifier | Dependency/backend | Key notes | Example settings in `model_params` |
| --- | --- | --- | --- | --- |
| Logistic regression | `logistic_regression` | scikit-learn | Linear probability classifier | `{"C": 1.0, "max_iter": 1000}` |
| Random forest | `random_forest` | scikit-learn | Tree ensemble | `{"n_estimators": 50, "max_depth": 5}` |
| Gradient boosting | `gradient_boosting` | scikit-learn | Boosted trees | `{"n_estimators": 50}` |
| LightGBM | `lightgbm` | Native LightGBM in pandas extra | sklearn-compatible wrapper | `{"n_estimators": 50, "learning_rate": 0.05}` |
| Custom classifier/pipeline | Estimator object | Cloneable sklearn API | Must implement `predict_proba` | Estimator or pipeline parameter names |

For a decision tree, pass `sklearn.tree.DecisionTreeClassifier(...)` as `model`;
`train_model` does **not** accept the string `"decision_tree"`, although pandas
selection/CV factories do. [Standalone training examples](docs/training.md#pandas-training-example)
show fitting and prediction. Scale-sensitive models need appropriately fitted
preprocessing; passing an sklearn Pipeline keeps learned preprocessing in its fit.

### pandas validate and optimize

**Continuing the main workflow.** Compare the baseline with one alternative on
May validation; do not look at June OOT while choosing. This small manual search
preserves the chronological split and the training-only selected subset.

```python
alternative_params = {"logisticregression__C": 1.0, "logisticregression__max_iter": 1000}
alternative = train_model(
    "alternative",
    "target",
    selected_features,
    train_df,
    validation_df,
    model=classifier,
    model_params=alternative_params,
    save_model_to="none",
)
best_params = (
    alternative_params
    if alternative.metrics["oot_auc"] > result.metrics["oot_auc"]
    else baseline_params
)
development_df = pd.concat([train_df, validation_df], ignore_index=True)
```

Freeze the feature list and chosen settings. The next step refits on development
training + validation and evaluates June once. CV instead estimates performance
across development folds; Optuna optimization evaluates multiple configurations.
See [verified CV and optimization alternatives](docs/journey-validation.md#pandas-api-alternatives)
for `cross_validate` / `optimize`, search spaces, and final refitting.
These alternatives use fixed candidate features, not selection fitted outside CV.
For learned preprocessing or supervised selection in CV, use a fold-fitted pipeline or explicit per-fold fitting; preselecting on all development rows biases CV.
The optimizer accepts no fold/preprocessing callbacks; its default CV is stratified.
Never pass the final OOT set as optimization's `oot_df`; pass development validation.

### pandas evaluate metrics

**Continuing:** this is the first model decision-independent use of June OOT.

```python
result = train_model(
    "final",
    "target",
    selected_features,
    development_df,
    oot_df,
    model=classifier,
    model_params=best_params,
    save_model_to="none",
)
predictions = result.oot_predictions
print(result.metrics)
```

Targets are binary 0/1. AUC/Gini expect higher scores for class 1; ranking scores
need not be calibrated probabilities. The model's `score` is a probability.
KS measures separation magnitude, not calibration or direction. Scalars are in
decimal units; Gini is `2 * AUC - 1`. Undefined metrics are NaN for single-class
or empty valid pairs (Spark grouped output may have no row for absent groups).

| Metric or utility | What it answers | Interpretation | Usage/example |
| --- | --- | --- | --- |
| `ks` | How separated are class distributions? | 0?1; larger means more separation; exact score thresholds. | [pandas KS](docs/evaluation.md#pandas-kolmogorov-smirnov-ks) |
| `auc_gini` | How well are positives ranked? | AUC 0.5 is chance ranking, 1 is perfect; Gini 0 and 1 respectively. | [AUC/Gini](docs/evaluation.md#pandas-roc-auc-and-gini) |
| `roc_auc_gini` | Same AUC/Gini calculation | Public alias of `auc_gini`, same signature/shape. | [Alias example](docs/evaluation.md#public-alias-and-scalar-utility) |
| `curve_roc` | Tradeoff across thresholds | DataFrame of curve points; plot separately. | [ROC example](docs/evaluation.md#pandas-plot-a-roc-curve) |
| `risk_table` | Event rate across ordered score tiles | DataFrame of occupied tiles, ranges, volumes, and event rates. | [Risk table](docs/evaluation.md#pandas-risk-sorting-table) |
| `calculate_ks` | Scalar KS from two Series | Float; no grouping wrapper. | [Scalar example](docs/evaluation.md#public-alias-and-scalar-utility) |

```python
from brainmodelkit.metrics.pandas import auc_gini, ks

print(ks(dataframe=predictions, group_columns="segment"))
print(auc_gini(df=predictions))
```

Pandas computes exact KS and sklearn ROC AUC/ROC. KS/AUC wrappers support grouping; filter the input for segment-specific ROC/risk tables. Tied score behavior differs from Spark tiling.
See [metric input and undefined-case details](docs/evaluation.md).

### pandas analyze segments

Overall performance can hide weaker segments. Start with a few meaningful,
adequately populated dimensions from `predictions`.

```python
from brainmodelkit.cube_analysis.pandas import calculate_metrics, calculate_ntile

cube = calculate_metrics(df=predictions, group_columns=["segment"])
tile_cube = calculate_ntile(df=predictions, group_columns=["segment"], n_tiles=4)
print(cube)
print(tile_cube)
```

Metric rows contain `segment`, `KS`, `AUC`, `Gini`, `score`; `score` here is the
**source score column name**. `Geral` denotes the overall rollup. With more
dimensions, omitted dimensions also contain `Geral`; actual missing keys remain
missing. Each score requires `2 ** dimensions` grouping calculations.
Risk cubes additionally report volumes/events per occupied tile; use these to
judge support. Small groups are noisy, single-class metrics are undefined, and
there is no automatic minimum-volume suppression. Real keys named `Geral` collide
visually with rollups. [Cube interpretation](docs/evaluation.md#interpreting-segment-results)
includes an illustrative result and full output conventions.

### pandas save, load, and score

The walkthrough used `save_model_to="none"`. To persist the frozen final recipe,
run that final fit with folder persistence enabled (or use `--save-dir` in the
[separate fixture-based runnable example](examples/journey/README.md)). This does not choose new settings.

```python
saved_result = train_model(
    "saved_final",
    "target",
    selected_features,
    development_df,
    oot_df,
    model=classifier,
    model_params=best_params,
    save_model_to="folder",
    model_format="pickle",
    output_dir="training_runs",
)
print(saved_result.output_dir, saved_result.model_uri)

from brainmodelkit.persistence import load_model

loaded = load_model(saved_result.model_uri, format="pickle")
new_data = oot_df.drop(columns="target")  # Replace with future rows of the same schema.
positive = list(loaded.classes_).index(1)
new_data["score"] = loaded.predict_proba(new_data[selected_features])[:, positive]
print(new_data[["account_id", "score"]])
```

Folder mode writes the model plus metadata, metrics, and feature-importance reports
in a unique run directory. Preserve the feature recipe, preprocessing policy, types,
and order; the persisted estimator does not reconstruct history or build features.
Pickle loads a sklearn-style estimator; only load trusted artifacts and keep the exact `selected_features` order.
MLflow uses its own model flavors; `none` writes no artifacts. Native LightGBM,
ONNX, and Spark loaders do not return the same object type or prediction interface.
See [persistence formats and loaders](docs/training.md#model-formats),
[MLflow](docs/training.md#mlflow), and [legacy migration](docs/training.md#migration-and-compatibility).

## Reference and troubleshooting

[Documentation home](docs/README.md) organizes the detailed guides by task.
The previous README's substantive material remains reachable through these topics;
[the move map](docs/navigation.md) records previous section destinations.

| Need | Destination |
| --- | --- |
| Installation, extras, Doctor, managed Spark, SynapseML, Windows | [Setup and troubleshooting](docs/setup.md) |
| Feature definitions, staleness, naming, simulation | [Temporal feature catalog](docs/temporal.md) |
| Selector signatures, rules, outputs, examples | [Selection index](docs/feature_selection.md) and [cost comparison](docs/selection-costs.md) |
| Model parameters, outputs, persistence, loaders, MLflow | [Training reference](docs/training.md) |
| CV folds, search spaces, scoring, constraints, result fields | [Validation](docs/model_selection.md), [optimization](docs/optimization.md), [safe alternatives](docs/journey-validation.md) |
| KS/AUC/Gini, ROC, risk tables, cube schemas | [Evaluation reference](docs/evaluation.md) |
| Complete prior demonstrations and feature-summary examples | [Standalone alternatives](docs/standalone-examples.md) |
| Version matrix and release checks | [Compatibility](docs/release-readiness.md) |
| Distributed costs and limitations | [Spark scalability](docs/setup.md#spark-scalability-limits) and [selector execution](docs/feature_selection/spark.md) |

<a id="migration-and-compatibility"></a>
Legacy training destinations/formats: [migration and compatibility](docs/training.md#migration-and-compatibility).
Legacy feature builders and date outputs: [feature migration](docs/temporal.md#migration).

<a id="training-parameter-reference"></a>
All training arguments: [parameter reference](docs/training.md#training-parameter-reference).

<a id="feature-selection"></a>
All sixteen selectors: [method index](docs/feature_selection.md#api-quick-reference).

<a id="numeric-time-series-features"></a>
Feature engineering: [Spark](#spark-build-features), [pandas](#pandas-build-features),
[full contract](docs/temporal.md).

## Project information

- [Development and contribution](CONTRIBUTING.md), [development commands and notebooks](docs/project.md#development).
- [Project layout](docs/project.md#project-layout), [changelog](CHANGELOG.md), [security policy](SECURITY.md).
- [MIT License](LICENSE).
