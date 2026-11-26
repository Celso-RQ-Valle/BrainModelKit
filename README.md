# BrainModelKit

BrainModelKit builds numeric time-series features, selects model inputs, trains
binary classifiers, and evaluates overall and segment performance. Choose the
backend that matches your DataFrames; the base install has no runtime dependencies.

After training, use `LogisticCalibrator` for event-probability calibration,
`ScoreAligner` for fixed bounded business scores, and `ModelExplainer` for
individual feature contributions. Both pandas and PySpark APIs are available.
See [calibration, scores, and explanations](docs/probability-workflow.md) for
formulas, supported models, execution limits, and runnable
[pandas](examples/probability_workflow_pandas.py) and
[Spark](examples/probability_workflow_pyspark.py) examples. The existing `score`
column remains a model probability; business scores use a separate column.

> [!WARNING]
> The public API is not stable yet. BrainModelKit is under active development
> and remains experimental, including its Spark integrations.

| Choose | Data and execution | Start here |
| --- | --- | --- |
| Spark | Spark DataFrames and native distributed operations; individual algorithms have driver/worker limits. | [Spark guide](#spark-guide) |
| pandas | In-memory pandas DataFrames and scikit-learn-compatible models. | [pandas guide](#pandas-guide) |

The journey is **install ? diagnose ? prepare ? build ? select ? validate/tune ?
train ? evaluate ? analyze segments ? save/load/score**. Each chapter
follows this order. Its snippets continue one workflow unless marked as an
alternative. [Complete runnable examples](examples/journey/README.md) include
imports, missing-value handling, and assertions using a separate small fixture.
The README workflow below uses the project's credit and bureau-history simulators,
also demonstrated in [the credit notebook](notebooks/simulate_credit_data.ipynb).

## Contents

- [Use a fitted or saved model](#use-a-fitted-or-saved-model)
  - [Local model workflow](#local-model-workflow)
  - [Native models, callables and optional MLflow](#native-models-callables-and-optional-mlflow)
  - [Model support](#model-support)
  - [Execution environments](#execution-environments)
- [Custom feature quickstart](#custom-feature-quickstart)
- [Spark guide](#spark-guide)
  - [Install and diagnose](#spark-install-and-diagnose)
  - [Prepare data](#spark-prepare-data)
  - [Build and select features](#spark-features)
  - [Validate and optimize](#spark-validate-and-optimize)
  - [Train](#spark-train)
  - [Evaluate metrics](#spark-evaluate-metrics)
  - [Analyze segments](#spark-analyze-segments)
  - [Calibration and score alignment](#spark-calibration-and-score-alignment)
  - [Explain predictions](#spark-explain-predictions)
  - [Save, load, and score](#spark-save-load-and-score)
- [pandas guide](#pandas-guide)
  - [Install and diagnose](#pandas-install-and-diagnose)
  - [Prepare data](#pandas-prepare-data)
  - [Build and select features](#pandas-features)
  - [Validate and optimize](#pandas-validate-and-optimize)
  - [Train](#pandas-train)
  - [Evaluate metrics](#pandas-evaluate-metrics)
  - [Analyze segments](#pandas-analyze-segments)
  - [Calibration and score alignment](#pandas-calibration-and-score-alignment)
  - [Explain predictions](#pandas-explain-predictions)
  - [Save, load, and score](#pandas-save-load-and-score)
- [External dataframe validation](#external-dataframe-validation)
- [Reference and troubleshooting](#reference-and-troubleshooting)
- [Project information](#project-information)

## Use a fitted or saved model

Choose the model source independently of its execution backend. Pass a fitted
estimator or `TrainingResult` directly, or use `ArtifactSpec(path=..., format=...)`
for a saved model. Training's `save_model_to` selects a destination and
`model_format` selects its representation. Loading uses the matching `format`.
Windows paths and mounted directories such as `/Volumes/...` work when visible
to the process. Use the MLflow format for `runs:/` and `models:/` URIs.
Spark model readers can also use filesystems configured in Spark.

For optional integrations, install the relevant extras, for example
`python -m pip install -e ".[pandas,onnx,xgboost,catboost,secure]"`.
The pandas and PySpark namespaces choose the DataFrame backend. Spark's
`execution="local"` and `"distributed"` choose where explanations run.
Driver, worker, JVM and filesystem capabilities depend on the actual runtime.

### Local model workflow

From a checkout, install the extras and run the [complete local example](examples/saved_workflow.py).

```bash
python -m pip install -e ".[pandas,persistence,explainability]"
python examples/saved_workflow.py
```

The direct pipeline below takes the in-memory training result. Its saved
counterpart recovers feature order and class labels, fits calibration on held-out
predictions, and explains the resulting score using the original training inputs.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from brainmodelkit.pandas import (
    LogisticCalibrator,
    ModelExplainer,
    PredictionPipeline,
    ScoreAligner,
)
from brainmodelkit.persistence import ArtifactSpec, load_workflow, save_workflow
from brainmodelkit.training.pandas import train_model


folder = TemporaryDirectory()
frame = pd.DataFrame({"income": np.arange(60) % 7, "debt": np.arange(60) % 5})
frame["target"] = ((frame.income > frame.debt) ^ (frame.index % 4 == 0)).astype(int)
features = ["income", "debt"]
train, calibration, oot = frame[:30], frame[30:45], frame[45:]
trained = train_model(
    "demo",
    "target",
    features,
    train,
    oot,
    model="logistic_regression",
    save_model_to="folder",
    model_format="joblib",
    output_dir=folder.name,
)
direct = PredictionPipeline(trained)
held_out = direct.transform(calibration)
calibrator = LogisticCalibrator().fit(held_out, "event_probability", "target")
artifact = ArtifactSpec(path=trained.model_uri, format="joblib")
workflow = PredictionPipeline(
    artifact, calibrator=calibrator, score_aligner=ScoreAligner(300, 850)
)
scored = workflow.transform(oot)
np.testing.assert_allclose(scored.event_probability, direct.predict(oot))
bundle = Path(folder.name) / "workflow"
save_workflow(bundle, workflow, format="joblib")
restored = load_workflow(bundle, format="joblib")
explanation = ModelExplainer(
    restored,
    train,
    target="business_score",
    method="permutation",
    max_background=8,
    max_evals=9,
)
result = explanation.explain(scored.head(2))
np.testing.assert_allclose(result.predictions, scored.business_score_unrounded.head(2))
print(result.to_frame())


folder.cleanup()
```

Use `feature_space="transformed"` when bundling an estimator fitted on already transformed coordinates. Supply those same inputs and names for explanations.
The bundle contains a relative model path, ordered schema when known, event
orientation, postprocessing configuration and package versions. Supply the
expected format when loading and only open trusted pickle-family artifacts.
To reference an existing artifact, pass `reference=artifact` when saving and
again when loading. Optional `background_reference={"uri": ..., "sha256": ...}`
records the identity of a separately managed reference dataset. Supply that
data to the explainer again. To embed an explicit pandas reference sample, pass
`background=sample` with PyArrow installed. Loading verifies its SHA-256 digest and
exposes `restored.background`. Only declared feature columns are embedded. Custom adapters require registration in each process.

### Native models, callables and optional MLflow

For a LightGBM native file, supply the original ordered class labels when the
training report is unavailable. This example continues from the local example's
`train`, `oot` and `features`.

```python
from lightgbm import Booster, LGBMClassifier

with TemporaryDirectory() as native_dir:
    path = Path(native_dir) / "credit.txt"
    fitted = LGBMClassifier(n_estimators=5, min_child_samples=2, verbosity=-1).fit(
        train[features], train.target
    )
    fitted.booster_.save_model(str(path))
    native = PredictionPipeline(
        ArtifactSpec(path=path, format="native", model_class=Booster),
        features,
        class_labels=list(fitted.classes_),
        event_label=1,
    )
    np.testing.assert_allclose(
        native.predict(oot), fitted.predict_proba(oot[features])[:, 1]
    )
```

For a custom probability function, return one probability of the declared event
per row. Callables require explicit feature_cols, including when passed directly
to ModelExplainer. Supply a separate `raw_callable` for an event-oriented raw target.

```python
def probability(inputs):
    return direct.predict(inputs)


custom = PredictionPipeline(None, features, prediction_callable=probability)
explained = ModelExplainer(
    custom,
    train,
    target="event_probability",
    method="permutation",
    max_background=8,
    max_evals=9,
)
print(explained.explain(oot.head(1)).to_frame())
```

MLflow is optional. Install `python -m pip install -e ".[pandas,mlflow]"` for
this alternative using the same fitted estimator. A registered URI may replace
the local MLflow directory.

```python
import mlflow.sklearn

with TemporaryDirectory() as mlflow_dir:
    path = Path(mlflow_dir) / "credit"
    mlflow.sklearn.save_model(trained.model, str(path), pip_requirements=[])
    tracked = PredictionPipeline(
        ArtifactSpec(path=path, format="mlflow", flavor="sklearn")
    )
    np.testing.assert_allclose(tracked.predict(oot), direct.predict(oot))
```

### Model support

| Fitted model | Loaded format | Prediction | Explanation method and target | Preprocessing and execution |
| --- | --- | --- | --- | --- |
| sklearn logistic regression | pickle, joblib, cloudpickle, trusted skops, MLflow sklearn | Probability and decision margin | Linear SHAP for raw, permutation for probability | pandas, fitted pipelines use permutation |
| sklearn random forest, decision tree, gradient boosting | Same sklearn formats | Probability, also decision margin for gradient boosting | Tree SHAP for base probability or uncalibrated points, gradient boosting also raw | Direct numeric inputs for optimized SHAP |
| Binary LightGBM classifier or Booster | Python formats, native Booster, MLflow lightgbm | Probability and margin | Interventional Tree SHAP for raw or base probability | Direct numeric inputs for optimized SHAP |
| Binary XGBoost classifier or logistic Booster, CatBoost classifier | Python formats, native, installed native MLflow flavor | Probability and margin | Explicit permutation | pandas, native Booster needs class order |
| Other binary classifiers, registered adapters, callables | Python formats where serializable, or in memory | Ordered two-column `predict_proba` or declared event callable | Explicit permutation for supported outputs | Select original inputs, retain fitted preprocessing |
| ONNX Runtime binary classifier | ONNX or MLflow ONNX | Selected probability tensor or ZipMap and explicit ordered classes | Explicit permutation for probability | Supported tensor dtypes, CPU session |
| MLflow pyfunc | MLflow, explicit pyfunc flavor | Explicit named event-probability DataFrame column or callable | Explicit permutation | Caller supplies original features and output meaning |
| Spark binary classifier or PipelineModel | Spark, MLflow Spark | Native transform and configured probability vector | Exported logistic or binary LightGBM only | Native preprocessing for prediction. Worker export accepts scalar inputs and a supported classifier, optionally after VectorAssembler |

All pandas probability adapters can apply calibration and alignment. Select
permutation explicitly for calibrated outputs and unrounded score points after
calibration. Rounded points are display values. For custom model classes,
register `register_adapter(ModelClass, factory)` from
`brainmodelkit.prediction.adapters`. The factory receives the fitted object and
event-class index. Its `probability(frame)` and `raw(frame)` methods return one
event-oriented value per row and should raise `CapabilityError` for unavailable
outputs. An optional `raw_units` attribute describes the margin units. Custom
registrations support local prediction and permutation explanation.

Spark prediction reads the fitted classifier's probability column. Pass
`probability_col="name"` for a pipeline that exposes its binary probability
vector under a different name. Standalone Spark ML classifiers require their
fitted vector column among the declared inputs.

### Execution environments

| Runtime | Status and restrictions |
| --- | --- |
| Local Python / pandas | Tested with small binary models and local artifacts. Install the model framework as well as the chosen persistence extras. |
| Classic local Spark | Tested native Spark ML prediction and bounded local diagnostics on Windows. In this environment, a Python worker running SHAP terminated in a minimal Arrow job. Distributed explanations require matching worker Python, SHAP, pandas, PyArrow and model dependencies. |
| SynapseML on classic Spark | Parity test supplied, unvalidated locally because the JVM assembly was unavailable. The adapter checks native text predictions against Spark probabilities on a bounded reference before explanation. |
| Spark Connect / Databricks Serverless | Unvalidated. SQL calibration and alignment have separate requirements from Spark ML fitting or JVM model export. |
| Arbitrary remote Python artifact URIs | Unsupported. Stage to a local or mounted path, or use a supported MLflow source. |

## Custom feature quickstart

This standalone example uses pandas and includes all its input data. From a
checkout containing the new API, install the backend:

```bash
python -m pip install -e ".[pandas]"
```

Copy and run the complete block below, or run
`python examples/temporal/custom_columns.py` from the repository root.
Two accounts share a customer ID; the composite key keeps their histories separate.

```python
import pandas as pd

from brainmodelkit.pandas import build_features

df_base = pd.DataFrame(
    {
        "customer_id": ["c1", "c1", "c1", "c1"],
        "account_id": ["a1", "a1", "a1", "a2"],
        "observation_date": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-03-01"],
        "used_balance": [10.0, None, 20.0, 90.0],
        "overdue_balance": [1.0, 2.0, None, 5.0],
        "credit_limit": [100.0, 0.0, 100.0, 200.0],
    }
)
df_spine = pd.DataFrame(
    {
        "customer_id": ["c1", "c1"],
        "account_id": ["a1", "a2"],
        "dt_ref": ["2026-03-01", "2026-03-01"],
    }
)
create_cols = {
    "total_balance": {
        "op": "sum",
        "args": [{"col": "used_balance"}, {"col": "overdue_balance"}],
    },
    "total_utilization": {
        "op": "divide",
        "args": [
            {
                "op": "sum",
                "args": [{"col": "used_balance"}, {"col": "overdue_balance"}],
            },
            {"col": "credit_limit"},
        ],
    },
}
# These options do not duplicate any explicitly passed keyword arguments.
feature_options = dict(windows=[3], lags=[1], period_unit="months")
enriched = build_features(
    df_base=df_base,
    df_spine=df_spine,
    key_cols=["customer_id", "account_id"],
    value_cols=["used_balance", "credit_limit", "total_balance", "total_utilization"],
    create_cols=create_cols,
    nullable=False,
    relative_historico=True,
    **feature_options,
)
default = build_features(
    df_base=df_base,
    df_spine=df_spine,
    key_cols=["customer_id", "account_id"],
    value_cols=["used_balance", "credit_limit"],
    create_cols=None,
    nullable=True,
    **feature_options,
)
assert enriched.loc[0, "used_balance_mean_3_months"] == 10
assert default.loc[0, "used_balance_mean_3_months"] == 15
print(
    enriched[
        [
            "customer_id",
            "account_id",
            "total_balance_mean_3_months",
            "total_utilization_mean_3_months",
        ]
    ].to_string(index=False)
)
```

Expected output (rounded here):

| customer_id | account_id | total_balance_mean_3_months | total_utilization_mean_3_months |
| --- | --- | ---: | ---: |
| c1 | a1 | 11.0 | 0.103333 |
| c1 | a2 | 95.0 | 0.475000 |

`sum` adds operands on each historical row. Both custom names must appear in
`value_cols`; `overdue_balance` is only an operand and needs no temporal outputs.
The nested utilization expression uses original columns, since custom expressions
cannot reference another newly declared custom column.

With `nullable=False`, the existing null observations become zero before arithmetic.
February's zero credit limit produces a null utilization, then zero before temporal
aggregation. No absent periods are added. The assertions demonstrate that the mean
of `[10, null, 20]` is 10 under this policy and 15 with the defaults.
`relative_historico=True` additionally compares each series with its own past.
For more options, see the [full parameter reference](docs/temporal.md).

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
        F.date_add(
            F.lit(date(2026, 4, 1)), F.datediff("reference_date", F.lit(first_date))
        ),
    )
    .withColumnRenamed("company_id", "account_id")
    .withColumnRenamed("default_flag", "target")
    .withColumnRenamed("company_size", "segment")
)
df_base = simulate_bureau_history(
    df_spine.select("account_id"),
    entity_cols="account_id",
    initial_date="2025-10-01",
    final_date="2026-06-30",
    seed=42,
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
    key_cols=["account_id"],
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

For explicit row-level arithmetic, use `create_cols` with nested operation
dictionaries; see the [runnable composite-key example](examples/temporal/custom_columns.py)
and [parameter reference](docs/temporal.md#custom-row-level-arithmetic).
Defaults `create_cols=None, nullable=True` preserve temporal values. Set
`nullable=False` to zero-fill existing numerical observations before calculations,
without creating absent periods. Generated feature names now use single
underscores; update downstream feature lists (see [migration](docs/temporal.md#migration)).

Enable comparisons with each variable's own history using the same inputs:

```python
feature_options = dict(
    key_cols=["account_id"],
    reference_date_col="reference_date",
    windows=[1, 3, 6],
    lags=[1, 2, 3],
    period_unit="months",
    aggregations=["mean", "median", "min", "max", "std", "sum", "count"],
)
enriched = build_features(
    df_base,
    df_spine,
    value_cols=["bureau_score"],
    relative_historico=True,
    **feature_options,
)
history_features = [c for c in enriched.columns if c not in df_spine.columns]
candidate_features = credit_features + history_features
```

For a utilization example, derive reproducible illustrative balances and changing
limits from the simulated history, then define a custom ratio expression.
Set `relative_historico=False` for
ratio-series temporal features alone.

```python
df_base = df_base.withColumn(
    "used_balance", F.col("bureau_score") * F.lit(0.2)
).withColumn("credit_limit", F.lit(100.0) + F.month("observation_date") * F.lit(20.0))
enriched = build_features(
    df_base,
    df_spine,
    value_cols=["used_balance", "credit_limit", "used_balance_over_credit_limit"],
    create_cols={
        "used_balance_over_credit_limit": {
            "op": "divide",
            "args": [{"col": "used_balance"}, {"col": "credit_limit"}],
        }
    },
    relative_historico=True,
    **feature_options,
)
history_features = [c for c in enriched.columns if c not in df_spine.columns]
candidate_features = credit_features + history_features
```

Each call replaces `enriched`; continue below with the version you want to model.
The last call includes original features, per-observation ratios, and historical
comparisons for each selected series. Used-balance/limit is utilization;
cross-column relationships require explicit expressions.
See [relationships and naming](docs/temporal.md#optional-relationship-families)
for formulas, ratio scales, and missing-value rules.

`date_col` defaults to `observation_date`. Windows are **(reference ? k periods,
reference]**; lags use the latest non-null value at or before reference ? k periods.
Units are days, seven-day weeks, or calendar months (clipped at month end).
Supported aggregations are mean, exact median, min, max, sample std, **sum**, and
count. Each source gets current/lag values, `_date_diff_days` latency relative
to the spine reference date, and `current_minus_mean` for every window.
No generated lookup dates are returned. The spine's columns and duplicate rows remain.

Missing lookups/statistics are null; empty windows have count zero, including a
null sum. This recipe uses fixed zero filling as an explicit demonstration policy;
it learns no statistics from validation or OOT. In production, fit any learned
imputer/encoder on training data only and inside folds for CV. Keys, dates, target,
segment, status, and industry section never enter `candidate_features`.
The inputs combine the simulator's numeric credit columns with all enabled temporal features.

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
all aggregations, optional relationships, staleness, boundaries, and validation.

<a id="build-feature-parameters"></a>
The same parameters apply to both backends:

| Parameter | Default | Effect |
| --- | --- | --- |
| `key_cols` | Required | Entity key column(s). |
| `value_cols` | Required | Numeric historical series to enrich. |
| `date_col` | `"observation_date"` | Observation date in `df_base`. |
| `reference_date_col` | `"dt_ref"` | Reference date in `df_spine`. |
| `period_unit` | `"months"` | Calendar months, weeks, or days. |
| `windows` | `(3, 6, 9)` | Rolling-window lengths. |
| `lags` | `(1, 2, 8)` | As-of lookup offsets. |
| `aggregations` | `("mean", "median", "max", "min", "std", "sum", "count")` | Requested window statistics. |
| `max_staleness_days` | `None` | Optional maximum age relative to each lookup cutoff. |
| `relative_historico` | `False` | Compare current values with requested lags and enabled level statistics. |
| `create_cols` | `None` | Named arithmetic expressions; include each name in `value_cols`. |
| `nullable` | `True` | Preserve missing observations; `False` zero-fills numeric observations. |

Current/lag values, observation ages, requested window statistics, and
`current_minus_mean` are automatic. Own-history comparisons require
`relative_historico=True`; cross-column ratios require `create_cols`.

**Migration:** `ref_col` has been removed on both backends. Calls such as
`ref_col="credit_limit"` must use an explicit divide expression as above and
include its name in `value_cols`. Percentage-point differences are no longer
automatically generated. See [migration](docs/temporal.md#migration).

Validate provider returns before generating features with
[external dataframe validation](#external-dataframe-validation).

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

### Spark validate and optimize

Tune logistic regression on April training and May validation, keeping June OOT
untouched. Install `brainmodelkit[pyspark,optimization]` for this step.
This example **maximizes validation AUC while limiting the training-to-validation KS
gap to five percentage points**. The API calls validation `oot_df`, so its
metrics have the `oot_` prefix. Hyperparameter lists specify `[minimum, maximum]` search
intervals or `[value]` for fixed settings. Omit an entry to keep its default.
See [models and search spaces](docs/models.md#spark-parameters) for every supported
model and its backend-specific parameters.

```python
from brainmodelkit.optimization.pyspark import optimize

search_space = {
    "regParam": [0.001, 1.0],  # Default if omitted: 0.0
    "maxIter": [100],  # Default if omitted: 100
}
tuning = optimize(
    train_df=train_df,
    oot_df=validation_df,
    target_col="target",
    feature_cols=selected_features,
    model="logistic_regression",
    search_space=search_space,
    objective_metric="oot_auc",
    direction="maximize",
    target_metric="train_oot_ks_abs_shift",
    target_limit=0.05,
    target_operator="<=",
    n_trials=20,
    cv=3,
    save_trials_path="optimization_trials.csv",  # Optional parameters/metrics export.
)
print(tuning.trials)  # Parameters, train/CV/validation metrics, and feasibility.
if not tuning.constraint_satisfied:
    raise ValueError("No feasible trial; review the search space or metric limit.")
best_params = tuning.best_params
```

The limit is a feasibility condition; AUC continues to improve among feasible
trials within the trial budget. To instead require KS of at least 0.40 while
improving AUC, use `target_metric="oot_ks"`, `target_limit=0.40`, and
`target_operator=">="`. Neither setting guarantees a feasible result or stops
at the first feasible trial. See [optimization](docs/optimization.md) for trial
fields, constraints, and the least-violating fallback.

The optimizer also computes stratified CV diagnostics on April; these are not
forward temporal validation. The fixed feature list here comes from a training-only
variance filter. Learned preprocessing and supervised selection must be fitted
inside folds for unbiased CV; the optimizer has no fold callbacks.
See [validation alternatives](docs/journey-validation.md#spark-api-alternatives).

### Spark train

Freeze the selected features and parameters, then refit on April + May.
June OOT is evaluated once, after tuning. Targets must be binary 0/1; `score`
is the fitted model's class-1 probability.

```python
from brainmodelkit.training.pyspark import train_model

development_df = train_df.unionByName(validation_df)
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
print(result.metrics)
```

`TrainingResult` contains the fitted model, metrics, feature importance,
`oot_predictions`, optional `scoring_predictions`, and persistence metadata.
Use the [full training contract](docs/training.md#training-parameter-reference)
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

### Spark evaluate metrics

Use the frozen model's June OOT predictions for the diagnostics below.

```python
predictions = result.oot_predictions
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
**source score column name**. `General` denotes the overall rollup. With more
dimensions, omitted dimensions also contain `General`; actual missing keys remain
missing. All `2 ** dimensions` grouping subsets are retained. Metric cubes expand rows
across scores and subsets before grouped aggregation; memory and work grow with
that product, so choose dimensions deliberately.
Risk cubes additionally report volumes/events per occupied tile; use these to
judge support. Small groups are noisy, single-class metrics are undefined, and
there is no automatic minimum-volume suppression. Real keys named `General` collide
visually with rollups. [Cube interpretation](docs/evaluation.md#interpreting-segment-results)
includes an illustrative result and full output conventions.

### Spark calibration and score alignment

Starting from `predictions = result.oot_predictions`, apply illustrative fixed
calibration coefficients, then map the calibrated default probability to a
300?850 business score. A lower default probability earns more points.

```python
from brainmodelkit.calibration.pyspark import LogisticCalibrator
from brainmodelkit.scoring.pyspark import ScoreAligner

calibrator = LogisticCalibrator(intercept=-0.2, slope=1.1)
aligner = ScoreAligner(300, 850)
calibrated = calibrator.transform(predictions, input_col="score")
scored = aligner.transform(calibrated, input_col="calibrated_probability")
scored.select("score", "calibrated_probability", "business_score").show(5)
```

To learn the coefficients, fit the calibrator on a separate labeled partition
held out from model training and tuning. The [runnable Spark example](examples/probability_workflow_pyspark.py)
shows the split and compares calibration before and after fitting.

### Spark explain predictions

Use `ModelExplainer` to see how each selected feature contributes to an individual
account's unrounded business score. Pass the calibrator and aligner from the
preceding step and use training rows as the reference for the explanation.

```python
from brainmodelkit.explainability.pyspark import ModelExplainer

explainer = ModelExplainer(
    result,
    train_df.select(*selected_features),
    selected_features,
    execution="local",
    max_rows=3,
    key_cols=["account_id", "reference_date"],
    calibrator=calibrator,
    score_aligner=aligner,
    target="business_score",
    method="permutation",
    max_background=20,
    max_evals=2 * len(selected_features) + 1,
)
explanation = explainer.explain(scored.limit(3))
print(explanation.to_frame())
```

See the [explainability reference](docs/probability-workflow.md) for optional
dependencies, supported models, and execution limits.

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

After the Spark folder fit above, load the saved pipeline and explain one
observation locally. Install `brainmodelkit[pyspark,explainability]` on the
driver and workers for distributed mode. Run the standalone equivalent with
`python examples/probability_workflow_pyspark.py --execution local --save-model`
on a Spark installation with a working folder writer. Omit `--save-model` for
an in-memory fit.

```python
from pyspark.ml import PipelineModel
from brainmodelkit.persistence import ArtifactSpec
from brainmodelkit.pyspark import ModelExplainer, PredictionPipeline

artifact = ArtifactSpec(
    path=saved_result.model_uri, format="spark", model_class=PipelineModel
)
workflow = PredictionPipeline(artifact, selected_features)
diagnostic = ModelExplainer(
    workflow,
    train_df,
    selected_features,
    key_cols=["account_id", "reference_date"],
    target="raw",
    execution="local",
    max_rows=1,
)
print(diagnostic.explain(oot_df.limit(1)).to_frame())
```

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
    "simulated_credit.csv",
    dtype={"account_id": "string"},
    parse_dates=["reference_date"],
)
credit_features = [f"feature_{i:02d}" for i in range(1, 21)]
df_base = simulate_bureau_history(
    df_spine[["account_id"]],
    entity_cols="account_id",
    initial_date="2025-10-01",
    final_date="2026-06-30",
    seed=42,
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
    key_cols=["account_id"],
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

For explicit row-level arithmetic, use `create_cols` with nested operation
dictionaries; see the [runnable composite-key example](examples/temporal/custom_columns.py)
and [parameter reference](docs/temporal.md#custom-row-level-arithmetic).
Defaults `create_cols=None, nullable=True` preserve temporal values. Set
`nullable=False` to zero-fill existing numerical observations before calculations,
without creating absent periods. Generated feature names now use single
underscores; update downstream feature lists (see [migration](docs/temporal.md#migration)).

Enable comparisons with each variable's own history using the same inputs:

```python
feature_options = dict(
    key_cols=["account_id"],
    reference_date_col="reference_date",
    windows=[1, 3, 6],
    lags=[1, 2, 3],
    period_unit="months",
    aggregations=["mean", "median", "min", "max", "std", "sum", "count"],
)
enriched = build_features(
    df_base,
    df_spine,
    value_cols=["bureau_score"],
    relative_historico=True,
    **feature_options,
)
history_features = [c for c in enriched.columns if c not in df_spine.columns]
candidate_features = credit_features + history_features
```

Define explicit ratios and enable historical comparisons on balances and limits
from the simulated history. For ratios alone, set `relative_historico=False`.

```python
df_base = df_base.assign(
    used_balance=df_base["bureau_score"] * 0.2,
    credit_limit=100.0 + pd.to_datetime(df_base["observation_date"]).dt.month * 20.0,
)
enriched = build_features(
    df_base,
    df_spine,
    value_cols=["used_balance", "credit_limit", "used_balance_over_credit_limit"],
    create_cols={
        "used_balance_over_credit_limit": {
            "op": "divide",
            "args": [{"col": "used_balance"}, {"col": "credit_limit"}],
        }
    },
    relative_historico=True,
    **feature_options,
)
history_features = [c for c in enriched.columns if c not in df_spine.columns]
candidate_features = credit_features + history_features
```

Continue with the last `enriched` result, or choose either earlier variant.
Both backends share the [parameter defaults](#build-feature-parameters),
[formulas, naming, temporal rules, and null handling](docs/temporal.md#optional-relationship-families).

Missing lookups/statistics are null; empty windows have count zero, including a
null sum. This recipe uses fixed zero filling as an explicit demonstration policy;
it learns no statistics from validation or OOT. In production, fit any learned
imputer/encoder on training data only and inside folds for CV. Keys, dates, target,
segment, status, and industry section never enter `candidate_features`.
The inputs combine the simulator's numeric credit columns with all enabled temporal features.

```python
enriched[candidate_features] = enriched[candidate_features].fillna(0)
ref = pd.to_datetime(enriched.reference_date)
train_df = enriched.loc[ref < "2026-05-01"].copy()
validation_df = enriched.loc[(ref >= "2026-05-01") & (ref < "2026-06-01")].copy()
oot_df = enriched.loc[ref >= "2026-06-01"].copy()
```

Pandas sorts history once, uses binary-search boundaries, and caches distinct entity/reference results. Window reductions and memory still grow with requested features and history.
The [full feature catalog](docs/temporal.md#temporal-semantics) covers naming,
all aggregations, optional relationships, staleness, boundaries, and validation.

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

### pandas validate and optimize

Tune logistic regression on April training and May validation, keeping June OOT
untouched. Install `brainmodelkit[pandas,optimization]` for this step.
This example **maximizes validation AUC while limiting the training-to-validation KS
gap to five percentage points**. The API calls validation `oot_df`, so its
metrics have the `oot_` prefix. Hyperparameter lists specify `[minimum, maximum]` search
intervals or `[value]` for fixed settings. Omit an entry to keep its default.
See [models and search spaces](docs/models.md#pandas-parameters) for every supported
model and its backend-specific parameters.

```python
from brainmodelkit.optimization.pandas import optimize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Scaling is learned separately inside each fit, including CV folds.
classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

search_space = {
    "logisticregression__C": [0.001, 10.0],  # Default if omitted: 1.0
}
tuning = optimize(
    train_df=train_df,
    oot_df=validation_df,
    target_col="target",
    feature_cols=selected_features,
    model=classifier,
    search_space=search_space,
    objective_metric="oot_auc",
    direction="maximize",
    target_metric="train_oot_ks_abs_shift",
    target_limit=0.05,
    target_operator="<=",
    n_trials=20,
    cv=3,
    save_trials_path="optimization_trials.csv",  # Optional parameters/metrics export.
)
print(tuning.trials)  # Parameters, train/CV/validation metrics, and feasibility.
if not tuning.constraint_satisfied:
    raise ValueError("No feasible trial; review the search space or metric limit.")
best_params = tuning.best_params
```

The limit is a feasibility condition; AUC continues to improve among feasible
trials within the trial budget. To instead require KS of at least 0.40 while
improving AUC, use `target_metric="oot_ks"`, `target_limit=0.40`, and
`target_operator=">="`. Neither setting guarantees a feasible result or stops
at the first feasible trial. See [optimization](docs/optimization.md) for trial
fields, constraints, and the least-violating fallback.

The optimizer also computes stratified CV diagnostics on April; these are not
forward temporal validation. The fixed feature list here comes from a training-only
variance filter. Learned preprocessing and supervised selection must be fitted
inside folds for unbiased CV; the optimizer has no fold callbacks.
See [validation alternatives](docs/journey-validation.md#pandas-api-alternatives).

### pandas train

Freeze the selected features and parameters, then refit on April + May.
June OOT is evaluated once, after tuning. Targets must be binary 0/1; `score`
is the fitted model's class-1 probability.

```python
from brainmodelkit.training.pandas import train_model

development_df = pd.concat([train_df, validation_df], ignore_index=True)
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
print(result.metrics)
```

`TrainingResult` contains the fitted model, metrics, feature importance,
`oot_predictions`, optional `scoring_predictions`, and persistence metadata.
Use the [full training contract](docs/training.md#training-parameter-reference)
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

### pandas evaluate metrics

Use the frozen model's June OOT predictions for the diagnostics below.

```python
predictions = result.oot_predictions
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
**source score column name**. `General` denotes the overall rollup. With more
dimensions, omitted dimensions also contain `General`; actual missing keys remain
missing. All `2 ** dimensions` grouping subsets are retained. Metric cubes expand rows
across scores and subsets before grouped aggregation; memory and work grow with
that product, so choose dimensions deliberately.
Risk cubes additionally report volumes/events per occupied tile; use these to
judge support. Small groups are noisy, single-class metrics are undefined, and
there is no automatic minimum-volume suppression. Real keys named `General` collide
visually with rollups. [Cube interpretation](docs/evaluation.md#interpreting-segment-results)
includes an illustrative result and full output conventions.

### pandas calibration and score alignment

Starting from `predictions = result.oot_predictions`, apply illustrative fixed
calibration coefficients, then map the calibrated default probability to a
300?850 business score. A lower default probability earns more points.

```python
from brainmodelkit.calibration.pandas import LogisticCalibrator
from brainmodelkit.scoring.pandas import ScoreAligner

calibrator = LogisticCalibrator(intercept=-0.2, slope=1.1)
aligner = ScoreAligner(300, 850)
calibrated = calibrator.transform(predictions, input_col="score")
scored = aligner.transform(calibrated, input_col="calibrated_probability")
print(scored[["score", "calibrated_probability", "business_score"]].head())
```

To learn the coefficients, fit the calibrator on a separate labeled partition
held out from model training and tuning. The [runnable pandas example](examples/probability_workflow_pandas.py)
shows the split and compares calibration before and after fitting.

### pandas explain predictions

Use `ModelExplainer` to see how each selected feature contributes to an individual
account's unrounded business score. Pass the calibrator and aligner from the
preceding step and use training rows as the reference for the explanation.

```python
from brainmodelkit.explainability.pandas import ModelExplainer

explainer = ModelExplainer(
    result,
    train_df[selected_features].head(20),
    selected_features,
    key_cols=["account_id", "reference_date"],
    calibrator=calibrator,
    score_aligner=aligner,
    target="business_score",
    method="permutation",
    max_background=20,
    max_evals=2 * len(selected_features) + 1,
)
explanation = explainer.explain(scored.head(3))
print(explanation.to_frame().to_string(index=False))
```

See the [explainability reference](docs/probability-workflow.md) for optional
dependencies, supported models, and execution limits.

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

## External dataframe validation

Use `brainmodelkit.pyspark.validate_external_dataframe` to check provider key
coverage, missing scores, monthly volume, baseline score-band stability and
fixed-bin score density. Install `matplotlib` for charts; notebooks display the
report automatically. The returned dictionary exposes metrics, aggregated tables,
thresholds, figures and warnings for reuse.

```python
from brainmodelkit.pyspark import validate_external_dataframe

report = validate_external_dataframe(
    df_sent=df_sent,
    df_received=df_received,
    key_cols=["customer_id", "reference_date"],
    date_col="reference_date",
    score_col="bureau_score",
)

report = validate_external_dataframe(
    df_sent=df_sent,
    df_received=df_received,
    key_cols=["customer_id", "reference_date"],
    date_col="reference_date",
    score_col="bureau_score",
    baseline="202601",
)
print(report["range_status"])
print(report["scores"]["below_range"], report["scores"]["above_range"])
print(report["scores"]["below_range_percentage"])
print(report["scores"]["above_range_percentage"])
```

The default baseline is the earliest valid received month, even if that month
has no valid scores. Coverage rates use distinct valid submitted/received keys
as separate denominators, excluding null-key rows and reporting duplicates
separately. All received rows still contribute to overall score analyses.

The stability chart freezes percentiles calculated only from finite, nonmissing
baseline scores in [0, 1000]. Deciles are [0, p10], (p10, p20], ..., (p90, 1000];
ties stay together and empty deciles remain visible. SQL null/NaN form Missing.
Out-of-range scores, including infinities, trigger a bold red notebook warning
(or labeled plain text) showing separate below-0 and above-1000 counts.
`report["scores"]["below_range_percentage"]` and `above_range_percentage` use all
received rows. Additional orange/red `Below 0` and `Above 1000` quality categories
keep monthly stacked percentages at 100%. Invalid dates are excluded only from
monthly analyses; nonnumeric scores are rejected. A baseline without in-range
scores makes deciles unavailable while preserving other report sections.
The histogram retains fixed 100-point bins over [0, 1000], including 1000,
normalized only over valid in-range scores, with excluded range counts reported.

See the [parameter and return reference](docs/external_validation.md), including
chart interpretation and Spark scalability, and run the
[complete simulated example](examples/external_validation.py) for three monthly
cohorts with missing/unexpected keys, duplicates, missing scores and a score shift.
Only aggregated summaries are collected to the driver.
