# Permutation Importance: permutation_importance_selection

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pandas as fs` for Pandas,
or `from brainmodelkit.feature_selection import pyspark as fs` for Spark.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

**Spark implementation:** read the [Spark algorithm and execution contract](spark.md#permutation_importance_selection).
The explanations below describe shared concepts and the Pandas behavior. Consult
the signatures for backend-specific arguments; they are not interchangeable.

## Signatures

### Pandas

```python
def permutation_importance_selection(
    model,
    df,
    target_col,
    feature_cols,
    *,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    threshold=None,
    top_k=None,
    n_jobs=None,
): ...
```

### PySpark

```python
def permutation_importance_selection(
    model,
    df,
    target_col,
    feature_cols,
    *,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    threshold=None,
    top_k=None,
): ...
```

The listings show accepted arguments and defaults. Pass optional controls by
keyword; permutation requires a fitted model first and RFE requires run_name first.
`None` placeholders for train/target/features do not make those inputs optional
at call time. Unsupported backend keywords raise TypeError.

## Parameters

| Name | Meaning |
| --- | --- |
| `model` | Named or compatible native estimator; permutation requires an already fitted model. Model restrictions and return schemas are explained below. |
| `df` | Alias for train_df except permutation, where this is the held-out development DataFrame. Supplying distinct frames under both aliases raises ValueError. |
| `target_col` | Target column name; exclude it from feature_cols. Required for supervised methods. Spark requires both 0 and 1; Pandas class support is described below. |
| `feature_cols` | Explicit nonempty ordered list of unique existing feature names. RFE alone can infer names starting feat_. Selection outputs preserve this order. |
| `scoring` | Pandas sklearn scorer; Spark accepts roc_auc or average_precision (Spark PR area, not sklearn average precision). |
| `n_repeats` | Positive permutation repetitions per feature; diagnostics report their mean and population standard deviation. |
| `random_state` | Seed for supported random estimators/resampling. Spark partition/order changes may still change results. |
| `threshold` | Inclusive score cutoff; None disables it. For correlation, reject at absolute correlation >= threshold in [0,1]. |
| `top_k` | Optional positive integer rank limit no larger than candidate count. Intersects other gates; ties favor input order. |
| `n_jobs` | Pandas sklearn parallel-job control; None uses the library default. Not accepted by Spark. |

## Returns and errors

Returns `SelectionResult`. Read `selected_features`, `rejected_features`, and
`feature_table`; method-specific fields are explained below. An empty selection
is valid and must be handled before fitting another model.

Invalid columns, duplicate names, empty inputs, incompatible targets, nonfinite
numeric inputs, or out-of-range controls raise validation errors. Quality methods
and IV have their documented missing-value exceptions. Optional model dependencies
raise installation errors when unavailable. Backend estimator errors, such as
invalid model parameters or inadequate CV class counts, are propagated.


## Example setup

Run the [shared example setup](setup.md) before the snippet below, or execute
`python examples/feature_selection/method_example.py --method permutation_importance_selection`
from the repository root for a self-contained Pandas example. Add `--backend spark`
for the Spark implementation. See the method-specific parameter tables above.

## How it works

**Supervised evaluation; Pandas behavior.** For an already fitted model, permute one
feature repeatedly and measure the decrease in a chosen score. Larger decreases
indicate stronger model reliance. This function does not refit the model.

```python
from brainmodelkit.feature_selection.pandas import permutation_importance_selection

result = permutation_importance_selection(
    model=fitted_model,
    df=oot_df,
    target_col="target",
    feature_cols=features,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    top_k=5,
)
```

Outputs are `feature`, `importance_mean`, `importance_std`, `ranking`, and
`selected`. `threshold` applies to the signed mean decrease, not its magnitude;
negative values can indicate noise. `n_jobs` controls sklearn parallel evaluation.
Use genuinely held-out data and preserve the fitted model's feature order.
If using that set to choose features, it becomes a selection set, not final test
data. Correlated features can substitute for one another and depress individual
importance. Repeats increase runtime; the standard deviation measures permutation
variation, not a confidence interval over new datasets. Spark supports native permutation; read the [Spark contract](spark.md#permutation_importance_selection).

## Spark usage

See [the complete Spark setup and permutation_importance_selection example](spark.md#permutation_importance_selection) for the
distributed algorithm, parameters, result fields, limits, and interpretation.
