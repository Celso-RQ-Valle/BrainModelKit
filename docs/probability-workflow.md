# Calibration, business scores, and individual explanations

[Documentation home](README.md) | [Training and persistence](training.md)

The workflow is **fitted model → event probability → calibrated event probability
→ business score → individual explanation**. Each component also works alone.
Existing training output `score` continues to mean the model's class-1 probability.
The new default display column is `business_score`; no existing training result or
serialized model format changes.

## Install and run

```bash
pip install 'brainmodelkit[pyspark,explainability]'
python examples/probability_workflow_pyspark.py --output spark_probability_artifacts

pip install 'brainmodelkit[pandas,explainability,plots]'
python examples/probability_workflow_pandas.py --output probability_artifacts
```

Spark additionally needs a working Java/Spark environment. Workers need the same
BrainModelKit, pandas, NumPy, SHAP, sklearn, LightGBM and PyArrow versions as the
driver. Plotting uses optional matplotlib. SHAP is imported only when an
explanation is initialized; the base package and fixed-parameter Spark SQL
transformations do not import SHAP or start Spark.

For local persistence and bundle round trips run `python examples/saved_workflow.py`.

Both scripts generate synthetic credit data, train on one partition, inspect a
separate development partition, fit calibration on held-out calibration
predictions, and evaluate only after freezing choices on a final partition.
They demonstrate manual parameters, fitted parameters, complementary
probabilities, 0–1000 and 300–850 scales, calibrated probability explanations,
score-point explanations, and portable JSON save/reload. Random splits in these
examples are holdouts, not chronological out-of-time validation.

## Public API

Use the same class names under either backend:

| Purpose | Spark namespace | Pandas namespace |
| --- | --- | --- |
| `LogisticCalibrator`, `calibration_diagnostics`, `compare_calibration` | `brainmodelkit.calibration.pyspark` | `brainmodelkit.calibration.pandas` |
| `ScoreAligner` | `brainmodelkit.scoring.pyspark` | `brainmodelkit.scoring.pandas` |
| `PredictionPipeline` | `brainmodelkit.prediction.pyspark` | `brainmodelkit.prediction.pandas` |
| `ModelExplainer`, `ExplanationResult` | `brainmodelkit.explainability.pyspark` | `brainmodelkit.explainability.pandas` |

The previous explainability imports of `PredictionPipeline` remain available.
The classes are also exposed lazily through `brainmodelkit.pyspark` and
`brainmodelkit.pandas`. An explainer or prediction composition accepts either a
fitted model or the existing `TrainingResult`; it unwraps `.model` without fitting
or modifying it. Native LightGBM loaders return a `Booster`: supply its original
`class_labels` explicitly (for repository training, `[0, 1]`).

## Logistic calibration

The sign convention is always:

```text
sigmoid(z) = 1 / (1 + exp(-z))
p_calibrated_event = sigmoid(intercept + slope * transformed_input)
```

| Configuration | Meaning |
| --- | --- |
| `input_space="logit"` (default) | Normalize to event probability, clip valid endpoints using epsilon, then use `log(p / (1-p))`. |
| `input_space="probability"` | Normalize to event probability and use `p` directly. |
| `input_space="raw"` | Use a finite model margin directly. Requires `input_probability=None`. |
| `input_probability="event"` (probability default) | Input already describes the configured event. |
| `input_probability="non_event"` | Normalize with `p_event = 1 - p_input` before any transformation. |
| `output_probability="event"` (default) | Return the calibrated event probability. |
| `output_probability="non_event"` | Return its complement, without refitting or changing coefficients. |
| `event_label=1` | The explicit target class called the event. A JSON string, boolean, or number is supported. |
| `epsilon=1e-12` | Must satisfy `0 < epsilon < 0.5`; used only for numerical stability of valid probabilities. |
| `invalid_policy="raise"` (default) | Reject null, NaN, infinity, invalid probabilities and nonnumeric input columns. |
| `invalid_policy="null"` | Map invalid numeric inputs to null/NaN; omit such rows during fitting/diagnostics. Nonnumeric columns still raise. |

`sigmoid(a + b*p)` and `sigmoid(a + b*logit(p))` are different functions. The
identity calibration `a=0, b=1` is an identity only in logit space, with endpoint
probabilities becoming epsilon and `1-epsilon`. Probabilities outside `[0,1]`
are invalid, never silently clipped into validity. Raw margins have a
model-defined orientation; complement options for raw input are rejected. The
composition adapters orient raw binary margins toward `event_label`.

### Spark

Import the Spark calibrator for use with Spark DataFrames.

```python
from brainmodelkit.calibration.pyspark import LogisticCalibrator

calibrated = LogisticCalibrator(intercept=-0.2, slope=1.1).transform(predictions, input_col="score")
```

Spark transformations are native SQL expressions, without Python UDFs, RDD access, or driver collection.
Strict validation executes a small existence query before returning the lazy
transformation. Output uses one projection against original columns, including
when explicit overwriting is requested.

### Pandas

```python
from brainmodelkit.calibration.pandas import LogisticCalibrator

manual = LogisticCalibrator(intercept=-0.2, slope=1.1)
calibrated = manual.transform(
    predictions,
    input_col="score",
    output_col="display_probability",
    event_col="event_probability",
    non_event_col="non_event_probability",
)

fitted = LogisticCalibrator(event_label=1).fit(
    calibration_predictions,
    input_col="score",
    target_col="default_flag",
    weight_col="sample_weight",  # Optional column, not a required weight.
)
print(fitted.intercept, fitted.slope, fitted.ranking, fitted.diagnostics_)
```

### Manual versus fitted parameters

Supplying both coefficients selects `mode="manual"`. No targets are needed,
and arbitrary coefficients are not evidence of empirical calibration. Calling
`fit` on a manual instance raises rather than replacing its parameters.
Omitting both coefficients selects `mode="unfitted"`; `fit` returns the same
instance with `mode="fitted"`. Transformation of an unfitted instance raises.

Fitting uses only the supplied prediction/label frame, never retrains the original
model, and cannot determine where that frame came from. Use held-out predictions
or out-of-fold predictions generated without training on each scored observation.
Reserve a distinct final evaluation set. Choosing a calibration scheme on that
final set would invalidate the separation.

Both backends minimize weighted, unpenalized binary logistic negative log
likelihood with an intercept (`regularization=0`). Pandas uses SciPy BFGS on a
centered/scaled predictor and converts coefficients back. Spark uses distributed
Spark ML binomial logistic regression with `regParam=0`, `standardization=True`,
and a single VectorAssembler input. Default `max_iter=1000`, `tol=1e-8`.
Only the small coefficient/diagnostic set is retrieved from Spark ML. Solver
convergence criteria differ; tests use coefficient tolerance `2e-5`, not bitwise
equality. If Spark ML fitting is unavailable, a `CapabilityError` retains the
underlying error; manual SQL transformations remain independently available.

Targets must contain exactly two classes including `event_label`. Fitting needs
at least three valid, positive-weight observations with both classes represented.
Weights must be finite, numeric, nonnegative; zero-weight rows are omitted.
Invalid weights always raise, including under the null policy. Null targets may
be omitted only under the null policy. Constant predictors and completely or
quasi-separated one-dimensional predictors raise because a unique finite
unpenalized slope cannot be estimated. Non-convergence raises; no regularization
or coefficient replacement is silently introduced. Diagnostics expose solver,
regularization, iterations, tolerance and used observations.

For probability inputs, `ranking` reports `preserves` for positive slope,
`reverses` for negative slope and `collapses` for zero slope, relative to normalized
event probability. Negative slopes are retained and reverse risk ordering;
inspect event definitions and data before using them. Requesting complementary
display probabilities does not alter this metadata or the underlying event model.

## Diagnostics

`calibration_diagnostics(frame, probability_col, target_col, ...)` returns a dict:

- `brier_score`, `log_loss`, `count`, `omitted_count`, `event_label`;
- `reliability`, a backend DataFrame with `bin`, `count`, `weight_sum`,
  `mean_probability`, `event_rate`.

Reliability bins are equal-width on `[0,1]`, numbered from zero; `p=1` belongs to
the last bin. Only nonempty bins are returned. Means and metrics use sample
weights when supplied; counts are positive-weight valid row counts. Log loss
uses epsilon clipping at valid endpoints. Spark uses distributed aggregation and
collects one scalar summary; the reliability table remains a Spark DataFrame.

`compare_calibration(frame, before_col, after_col, target_col, ...)` returns
`{"before": report, "after": report}`. Set `before_probability` and
`after_probability` independently to `event` or `non_event`. Other diagnostics
options apply to both reports. With the null policy, columns may have different
valid populations: compare counts and explicitly select a common population
before comparing metrics. Brier score and log loss reflect more than calibration;
they are not pure calibration measures. Improved metrics on unseen data are
never guaranteed.

## Linear probability-to-score alignment

This is a display scale, not a second calibration fit or an odds/PDO scorecard.
It does not improve discrimination. No fitting, batch minimum or maximum, or
batch-specific stretching is involved.

```text
r = score_max - score_min
p_event = p_input if input_probability == "event" else 1 - p_input

higher_score_means="lower_event_probability": score = score_min + r*(1-p_event)
higher_score_means="higher_event_probability": score = score_min + r*p_event
```

Defaults are `score_min=0`, `score_max=1000`, event input, and higher scores
meaning lower event probability. The endpoints are theoretical scale endpoints.
Event probabilities `0`, `0.2`, `1` map to `1000`, `800`, `0`. Non-event probability
`0.8` also maps to `800`. Every batch uses the same mapping.

```python
from brainmodelkit.scoring.pandas import ScoreAligner

aligner = ScoreAligner(score_min=300, score_max=850, decimals=0)
scored = aligner.transform(calibrated, input_col="event_probability")
recovered = aligner.inverse_transform(
    scored, input_col="business_score_unrounded", output_col="recovered_event"
)
```

Spark uses the same interface from `brainmodelkit.scoring.pyspark`, implemented
with SQL expressions. Finite bounds with `score_min < score_max` and finite range
are required. Validation and null handling match calibration. Pandas inputs and
their indexes are preserved; Spark preserves columns and keys, not row ordering.

Full precision is the default (`decimals=None`). Explicit `decimals=0..12` uses
decimal half-even rounding (Spark `bround` / pandas Decimal). Bounds must be
representable at the selected decimal precision. Rounded output retains
`business_score_unrounded` (or `unrounded_col`). Floating-point clipping protects
the theoretical bounds; no observation-based clipping or scaling is applied.
The inverse of unrounded scores is linear; inversion of rounded scores loses
information. Select `output_probability="non_event"` for the inverse complement.

All transforms accept output names and reject collisions unless
`overwrite=True`. Auxiliary outputs must have distinct names. To retain original
probability and both calibrated meanings, use separate columns. In composition,
event meaning is explicit: changing a probability display preference never flips
the score, even when standalone components accept non-event inputs.

## Compose predictions and explain the complete function

```python
from brainmodelkit.pandas import ModelExplainer, PredictionPipeline

workflow = PredictionPipeline(
    trained_result,
    feature_cols=features,
    event_label=1,
    calibrator=fitted,
    score_aligner=aligner,
)
predictions = workflow.transform(final_data)
explainer = ModelExplainer(
    workflow,
    background=reference[features],
    key_cols=["customer_id", "month"],
    target="business_score",
    method="permutation",  # Explicit cost/approximation choice.
    max_background=100,
    max_rows=50,
    max_evals=501,
    seed=42,
    reference_id="training-reference-v1",
)
result = explainer.explain(predictions.head(10))
print(result.summary(observation=0))
print(result.top_contributors(observation=0, top_k=3))
result.waterfall(observation=0, top_k=5)
```

`PredictionPipeline.transform` adds `event_probability`, optional
`calibrated_event_probability` / `calibrated_non_event_probability`, and optional
`business_score_unrounded` / `business_score`. All original columns remain. No
calibrator is required to score or explain the base model. An explainer does not
require a score aligner unless `target="business_score"`.

Pandas `PredictionPipeline.predict(frame, target)` selects the declared features
in model order. Supply `feature_cols` when the model has no input names.
A selected `prediction_callable` must return one **event probability**
per row, including its preprocessing; `raw_callable` supplies a separately
declared event-oriented margin. Custom callables require explicit permutation
SHAP. No model is inferred from its name. The adapters inspect actual types and
resolve `classes_`/explicit class labels; event class zero is not treated as
probability column one.

### Targets and mathematical correctness

| `target` | Units and explained function |
| --- | --- |
| `raw` | Base event-oriented model margin; log-odds for default logistic/binary LightGBM. |
| `event_probability` | Base model event probability, from 0 to 1. |
| `calibrated_event_probability` | Model + preprocessing + calibrator, in event probability units. |
| `calibrated_non_event_probability` | Complement of the preceding complete function. |
| `business_score` | Final **unrounded** score; uses calibrated probability when a calibrator is supplied. |

Probability contributions of `0.01` mean one percentage point. Raw log-odds and
score points are different units and cannot be compared directly. Nonlinear
calibration is part of the prediction callable; sigmoid is never applied
separately to contributions. Optimized explanations of a base model are never
relabeled as calibrated output.

For event-probability attributions `p = base + sum(phi)`, complementary output
uses `base'=1-base`, `phi'=-phi`. Higher-is-better score output uses
`base'=score_max-range*base`, `phi'=-range*phi`. These affine conversions apply
only after explaining the correct final event probability. Every result checks
`abs(prediction - base - sum(phi)) <= atol + rtol*abs(prediction)`; both tolerances
default to `1e-6`. Failure raises. Rounded scores are shown separately with their
rounding difference; contributions do not claim additivity to rounded output.

### Background, schema, and costs

Background is required and represents the reference population. Pandas samples
it once, deterministically with the seed, to at most `max_background=100` rows.
Spark selects a bounded deterministic content-hash sample before collecting that
reference. Metadata records identifier, content fingerprint, feature order,
dtypes, sample count, and seed. Keep the same explainer/reference across clients
and batches. Persist the reference frame alongside the configuration if it must
be reconstructed later; a fingerprint is not the reference data itself.

Prediction and explanation select the declared features in order from wider
frames. Missing features, duplicate column names, invalid observation keys, and
incompatible explanation dtypes raise. Spark workflows with saved input dtypes
check each declared input against its training type before native prediction.
pandas permutation masking supports
numeric features and scalar categories with fitted preprocessing, including
missing inputs when the estimator accepts them. Spark export and optimized
explanation use finite numeric model inputs. Optimized adapters require direct
model inputs; they do not silently label transformed coordinates as original
features. If explaining already transformed inputs, explicitly supply those
columns, names and reference and interpret them as that schema.

`method="optimized"` is the default and raises when unsupported.
`method="permutation"` is explicitly selected; there is no automatic expensive
fallback. `max_evals` must be at least `2*feature_count+1`. Costs scale with rows,
reference size, feature count, and permutation budget. `max_rows` bounds pandas
and explicit Spark local batches; in distributed mode it bounds worker chunks,
not the total distributed dataset. Model/explainer state is reused within each
partition. Permutation attributions are approximations whose allocations may
vary with seed and execution batching, while reconstruction remains checked.
Interventional background masking can create implausible combinations of
correlated features; attribution is not causal evidence or a suggested action.

### Results and reason codes

Pandas `ExplanationResult` retains predictions, base values, all contributions,
feature values, keys, model/calibration/scoring metadata, reference information,
target/units, approximation settings, and per-row residuals/tolerances.
`to_frame()` returns keys plus `prediction`, `base_value`, `residual`, `tolerance`,
`reconstruction_ok`, and `contribution::<feature>` / `value::<feature>` pairs.
Rounded score explanations also include `rounded_prediction` and
`rounding_difference`. Index identity is retained, even with duplicate indexes.

`top_contributors` returns separate positive and negative contributions.
`global_importance()` reports mean absolute contributions for this population
and output unit; it is not an individual explanation. `aggregate_groups()` sums
attributions without dropping any features. A `feature_dictionary` maps feature
names to optional `display_name`, `description`, `units`, and `group` fields.
Waterfalls show an **Other features** contribution when truncated. Deterministic
text summaries state observed values and contribution direction; positive SHAP
alone does not justify calling a value high, low, causal, or actionable. No LLM
or external API is needed.

### Spark execution

Spark `ModelExplainer` accepts the fitted model/TrainingResult plus explicit
`feature_cols`, `key_cols`, and calibration/scoring options. Its default
`execution="distributed"` uses `mapInPandas`; it exports coefficients/native
LightGBM text on the driver and reconstructs reusable local predictors inside
workers. It never sends JVM models into workers or invokes nested Spark jobs.
Input/output rows align by unique non-null composite keys, not ordering. Duplicate
keys raise. Validation executes distributed queries; only bounded background
rows are collected. Result contributions remain a Spark DataFrame.

Spark results support `to_frame`, distributed `global_importance`,
`top_contributors(top_k)` and `aggregate_groups`. For plotting or text, explicitly
filter the result to a customer and call `local(max_rows=1)`; it checks the bound
before converting to the pandas result. Alternatively select
`execution="local", max_rows=...` on the explainer. That mode is a bounded
driver diagnostic, not distributed explanation. Spark frames are lazy; cache
results when reusing expensive explanations across several actions.

### Model and output support

| Actual model/backend | Optimized raw | Optimized base probability / score without calibration | Calibrated event/complement/score | Preprocessing |
| --- | --- | --- | --- | --- |
| Spark binary `LogisticRegressionModel` | Exported linear SHAP | Explicit permutation | Explicit permutation | Matching scalar columns, optionally repository VectorAssembler pipeline |
| Spark BrainModelKit `SparkLightGBMModel` | Exported native Tree SHAP | Exported interventional Tree SHAP | Explicit permutation | Matching numeric columns, optionally repository VectorAssembler pipeline |
| Spark SynapseML binary LightGBM | Native Tree SHAP | Interventional Tree SHAP | Explicit permutation | Requires matching JVM runtime and bounded export parity check |
| Arbitrary Spark ML pipelines, multiclass, native categorical trees | Not implemented | Not implemented | Not implemented | Raise capability/schema errors |
| pandas sklearn `LogisticRegression` | Linear SHAP | Explicit permutation | Explicit permutation | Direct numeric features; sklearn Pipeline via permutation |
| pandas `LGBMClassifier` / binary native `Booster` | Tree SHAP | Interventional Tree SHAP | Explicit permutation | Direct numeric features |
| pandas sklearn decision tree / random forest / gradient boosting | Gradient boosting margin only | Interventional Tree SHAP | Explicit permutation | Direct numeric features |
| pandas other binary `predict_proba` or explicit callable | Only via explicit raw callable/permutation | Explicit permutation | Explicit permutation | Complete callable includes preprocessing |

Calibration/scoring can still process probability columns from any model,
including SynapseML. Explanation support does not imply fitting support. Spark
Connect and Databricks Serverless have **not** been validated; manual SQL
transformations avoid classic-only APIs, but Spark ML fitting and model export
have runtime-specific capabilities. Do not infer universal Serverless support.

## Save, reload, and portability

```python
fitted.save("calibration.json")
aligner.save("scoring.json")

from brainmodelkit.calibration.pyspark import LogisticCalibrator as SparkCalibrator
from brainmodelkit.scoring.pyspark import ScoreAligner as SparkAligner

spark_calibrator = SparkCalibrator.load("calibration.json")
spark_aligner = SparkAligner.from_dict(aligner.to_dict())
```

JSON has a strict schema version (`1`), kind, package version and configuration.
It includes intercept/slope, manual/fitted mode and diagnostics, input space,
event definition, input/output meanings, epsilon, invalid policy, feature order,
score bounds/direction and rounding precision. Supply `feature_order` when
constructing standalone configurations; composition validates it when present.
Fixed parameters are portable between Spark and pandas; no refitting is needed.
JSON does not contain the original estimator or background dataset. Save the
model with the existing [persistence APIs](training.md#saving-trained-models),
save the reference separately, and compose the loaded model/configurations.
Existing `TrainingResult` construction and serialized models are unchanged.

For artifact resolution, backend and runtime matrices, local and referenced bundles, and adapter registration, see the [saved-model walkthrough](../README.md#use-a-fitted-or-saved-model).
