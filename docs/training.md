# Training, model formats, and persistence reference

[Documentation home](README.md) ? [Modeling journeys](../README.md)

These retained examples are **standalone alternatives**, not consecutive cells.
Define the inputs named in each example. A random split called `oot` in an API
example is a holdout, not a chronological out-of-time evaluation. Never use a
final OOT period to select features or tune configurations.

- [Spark training module](#spark-training-module)
- [Pandas training module](#pandas-training-module)
- [Model Training](#model-training)

## Spark training module

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
probability. See the [full parameter reference](../docs/training.md#training-parameter-reference),
[training outputs](../docs/training.md#training-outputs), and [persistence options](../docs/training.md#saving-trained-models).

### Spark training example

Follow the [PySpark installation requirements](../docs/setup.md#installation) first.

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

#### Choosing the LightGBM backend

Pass `lgbm_backend` directly to `train_model()`, outside `model_params`.
`"default"` selects SparkLightGBM and is used when omitted; `"synapseml"`
explicitly selects SynapseML. See [Installation](../docs/setup.md#installation) and
[SynapseML setup](../docs/setup.md#synapseml-setup) for prerequisites.

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
[backend runtime limitations](../docs/optimization.md#spark-lightgbm) for worker
memory and barrier execution requirements.

The SparkLightGBM adapter supports raw-feature pipelines, class-1 probabilities,
and Spark folder/MLflow persistence. Loading requires BrainModelKit and the
selected LightGBM backend in the target environment. Configure persistence
before choosing `save_model_to="folder"` or `"mlflow"` with either backend;
see [Spark on Windows](../docs/setup.md#spark-on-windows).

To execute the standalone example from the repository root:

```bash
python examples/training/spark_example.py
```

## Pandas training module

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
probability. See the [full parameter reference](../docs/training.md#training-parameter-reference),
[training outputs](../docs/training.md#training-outputs), and [persistence options](../docs/training.md#saving-trained-models).

### Pandas training example

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

## Model Training

Start with the [Spark training module](../docs/training.md#spark-training-module) or [Pandas training module](../docs/training.md#pandas-training-module).

### Overview

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

### Supported Models

Both backends accept `model="logistic_regression"` (default), `"random_forest"`,
`"gradient_boosting"`, or `"lightgbm"`. Pass algorithm parameters in
`model_params` using the backend's native names, as shown above. You may also
pass a custom estimator instance through `model`; it must support probability
predictions. LightGBM is included in the Pandas extra.

Training labels must contain both classes, 0 and 1, and the target must not be
included in `feature_cols`. Preprocess your features before training. The examples
use random holdouts for demonstration; replace `oot_df` with a later-period
dataset for actual out-of-time evaluation.

### Saving Trained Models

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

#### Folder

##### How to execute `run_as="local"`

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

#### MLflow

##### How to execute `run_as="mlflow"`

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

##### Databricks Serverless / Shared Compute

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

#### No Persistence

##### How to execute `run_as="none"`

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

### Model Formats

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

### Training Outputs

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

### Loading Trained Models

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

### Advanced Configuration

Pass backend-specific estimator options through `model_params` and optional
unlabeled data through `df_scoring`. Pandas custom estimators must be cloneable
and implement `predict_proba`; Spark estimators must expose standard probabilistic
classifier columns. Inputs are not modified.

`signature=True` applies only to MLflow and describes native prediction labels, not the
added score column. Spark `n_tiles` controls metric approximation and must be at
least 2. Spark `overwrite` is forwarded to its native writer; automatic run
directories remain unique.

### Training parameter reference

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
| `lgbm_backend` (Spark-only) | `"default"` | LightGBM backend: `"default"` (SparkLightGBM) or `"synapseml"`. Pass directly to `train_model()`, outside `model_params`; see [backend examples](../docs/training.md#choosing-the-lightgbm-backend). |
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

### Migration and compatibility

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

## Verified differences between public entry points

Pandas `train_model` has named logistic regression, random forest, gradient
boosting, and LightGBM. Decision trees require an estimator object there, although
pandas selection and CV accept `"decision_tree"`. Spark training accepts that
identifier. Doctor's `model` choices omit decision trees and Spark's `"lgbm"`
alias: diagnose with `model=None` or the canonical supported name instead.

The Spark RFE docstring mentions SynapseML as required for LightGBM, but RFE
calls the shared trainer whose current default is SparkLightGBM. RFE does not
expose `lgbm_backend` itself; do not copy a training keyword into its call.
These are entry-point/documentation differences; this documentation reorganization
does not change behavior or public signatures.
