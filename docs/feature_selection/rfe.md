# RFE: rfe

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pandas as fs` for Pandas,
or `from brainmodelkit.feature_selection import pyspark as fs` for Spark.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

## Signatures

### Pandas

```python
def rfe(
    run_name,
    target_col,
    train_df,
    oot_df=None,
    feature_cols=None,
    model="logistic_regression",
    *,
    df_oot=None,
    step=1,
    n_feat_final=1,
    model_params=None,
    output_dir="feature_selection_runs",
    df_scoring=None,
    mlflow_logging=False,
    signature=False,
): ...
```

### PySpark

```python
def rfe(
    run_name,
    target_col,
    train_df,
    oot_df=None,
    feature_cols=None,
    model="logistic_regression",
    *,
    df_oot=None,
    step=1,
    n_feat_final=1,
    model_params=None,
    output_dir="feature_selection_runs",
    df_scoring=None,
    mlflow_logging=False,
    signature=False,
    n_tiles=10,
): ...
```

The listings show accepted arguments and defaults. Pass optional controls by
keyword; permutation requires a fitted model first and RFE requires run_name first.
`None` placeholders for train/target/features do not make those inputs optional
at call time. Unsupported backend keywords raise TypeError.

## Parameters

| Name | Meaning |
| --- | --- |
| `run_name` | Label used for final training outputs or RFE iteration reports. |
| `target_col` | Target column name; exclude it from feature_cols. Required for supervised methods. Spark requires both 0 and 1; Pandas class support is described below. |
| `train_df` | Training DataFrame for the selected backend. Fit selection on development/training rows only. |
| `oot_df` | Optional labeled OOT DataFrame for supported importance/wrapper final evaluation; required by historical RFE. It never chooses the feature subset. |
| `feature_cols` | Explicit nonempty ordered list of unique existing feature names. RFE alone can infer names starting feat_. Selection outputs preserve this order. |
| `model` | Named or compatible native estimator; permutation requires an already fitted model. Model restrictions and return schemas are explained below. |
| `df_oot` | Alias for oot_df. Supplying distinct frames under both spellings raises ValueError. |
| `step` | Positive integer number of weakest features removed per RFE/RFECV step; final removal is clipped to the minimum. |
| `n_feat_final` | RFE's required final count, integer between 1 and initial feature count. OOT metrics do not optimize this count. |
| `model_params` | Optional dictionary of native estimator parameters: Pandas snake_case, Spark camelCase. Custom estimators are copied/cloned. |
| `output_dir` | Root of RFE's unique run directory, with per-fit reports and history.json/selected_features.json. RFE has filesystem side effects. |
| `df_scoring` | Optional unlabeled frame to score with the final selected-feature classifier; requires OOT evaluation. RFE scores it only at the final fit. |
| `mlflow_logging` | Historical RFE persistence toggle; True sends model persistence to MLflow, False uses local folders. Requires configured optional MLflow when enabled. |
| `signature` | Whether to request an MLflow model signature for RFE persistence; requires the optional dependency. |
| `n_tiles` | Spark training/RFE metric approximation resolution; integer >=2. |

## Returns and errors

Returns `RFEResult`, containing `selected_features`, `rejected_features`, the final
`training_result`, per-iteration `history`, and a unique `output_dir`.

Invalid columns, duplicate names, empty inputs, incompatible targets, nonfinite
numeric inputs, or out-of-range controls raise validation errors. Quality methods
and IV have their documented missing-value exceptions. Optional model dependencies
raise installation errors when unavailable. Backend estimator errors, such as
invalid model parameters or inadequate CV class counts, are propagated.


## Example setup

Run the [shared example setup](setup.md) before the snippet below, or execute
`python examples/feature_selection/method_example.py --method rfe`
from the repository root for a self-contained Pandas example. Add `--backend spark`
for the Spark implementation. See the method-specific parameter tables above.

## How it works

**Supervised; Pandas and native Spark.** The original RFE API and implementation
remain available at both backend import paths. RFE is the recursive elimination
mechanism: fit, calculate native importance, remove the weakest features, refit,
and repeat until `n_feat_final` remain. The final subset is fitted and evaluated.

```python
from brainmodelkit.feature_selection.pandas import rfe

result = rfe(
    "credit_rfe",
    "target",
    train_df,
    oot_df,
    feature_cols=features,
    model="random_forest",
    model_params={"n_estimators": 50, "random_state": 42},
    step=2,
    n_feat_final=3,
    output_dir="feature_selection_runs",
)
print(result.history)
print(result.training_result.metrics)
```

Existing parameters include `run_name`, `target_col`, `train_df`, `oot_df`,
`feature_cols`, `model`, `model_params`, `step`, `n_feat_final`, `output_dir`,
`df_scoring`, `mlflow_logging`, and `signature`; Spark also has `n_tiles`.
Omitting features selects `feat_` columns. `step` and final count are positive
integers. The final count cannot exceed the initial count; equal counts fit once.
The existing supported model names remain unchanged; decision trees can be passed
as compatible estimator instances. Custom models need one finite importance per
input feature. RFE ranks absolute coefficients or native tree importance, removes
earlier input columns first on equal importance, and preserves retained order.

`RFEResult` still exposes `selected_features`, `training_result`, `history`, and
`output_dir`. History retains feature subsets, removed features, OOT KS/AUC/Gini,
report paths, and run IDs. Every iteration writes a training run under the unique
RFE directory; `history.json` and `selected_features.json` remain available.
Only the final model is retained and only the final fit scores `df_scoring`.
MLflow logging remains opt-in through the existing flag. Spark keeps frames
distributed; callers manage their own caches and Spark metrics use `n_tiles`.

RFE does not use OOT metrics to choose a subset or stop early. Users may
intentionally inspect a path such as `100 features → OOT KS`,
`90 → OOT KS`, `80 → OOT KS`, `70 → OOT KS`, and choose a stable region based on
external temporal behavior. That is a distinct selection procedure from RFECV;
repeated OOT inspection also consumes validation information. Reserve a final
untouched test period. RFE is useful when refitting reveals conditional importance,
but can be costly and inherits the model's importance biases and scaling issues.



