# Sequential Feature Selection: sequential_selection

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pyspark as fs` for Spark,
or `from brainmodelkit.feature_selection import pandas as fs` for pandas.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

**Spark implementation:** read the [Spark algorithm and execution contract](spark.md#sequential_selection).
The explanations below describe shared concepts and the Pandas behavior. Consult
the signatures for backend-specific arguments; they are not interchangeable.

## Signatures

### PySpark

```python
def sequential_selection(
    df,
    target_col=None,
    feature_cols=None,
    *,
    model="logistic_regression",
    model_params=None,
    direction="forward",
    n_features_to_select=1,
    scoring="roc_auc",
    cv=5,
    fold_col=None,
    random_state=42,
): ...
```

### Pandas

```python
def sequential_selection(
    df_train,
    target_col=None,
    feature_cols=None,
    *,
    df_oot=None,
    df_scoring=None,
    run_name="feature_selection",
    model="logistic_regression",
    model_params=None,
    direction="forward",
    n_features_to_select=1,
    scoring="roc_auc",
    cv=5,
    n_jobs=None,
    random_state=42,
): ...
```

The listings show accepted arguments and defaults. Pass optional controls by
keyword; permutation requires a fitted model first and RFE requires run_name first.

## Parameters

| Name | Meaning |
| --- | --- |
| `df` (Spark) or `df_train` (pandas) | Training DataFrame for the selected backend. Fit selection on development/training rows only. |
| `target_col` | Target column name; exclude it from feature_cols. Required for supervised methods. Spark requires both 0 and 1; Pandas class support is described below. |
| `feature_cols` | Explicit nonempty ordered list of unique existing feature names. RFE alone can infer names starting feat_. Selection outputs preserve this order. |
| `df_oot` | Optional labeled validation frame for final evaluation in pandas. |
| `df_scoring` | Optional unlabeled frame to score with the final selected-feature classifier; requires OOT evaluation. RFE scores it only at the final fit. |
| `run_name` | Label used for final training outputs or RFE iteration reports. |
| `model` | Named or compatible native estimator; permutation requires an already fitted model. Model restrictions and return schemas are explained below. |
| `model_params` | Optional dictionary of native estimator parameters: Spark camelCase, pandas snake_case. Custom estimators are copied/cloned. |
| `direction` | Sequential direction: forward adds one feature each round; backward removes one. |
| `n_features_to_select` | Sequential selection's desired positive feature count, strictly smaller than initial count. |
| `scoring` | Spark accepts roc_auc or average_precision (Spark PR area). Pandas accepts a sklearn scorer. |
| `cv` | Spark takes an integer >=2 and requires both classes in every validation fold. Pandas accepts a fold count or sklearn-compatible splitter/iterable. |
| `n_jobs` | Pandas sklearn parallel-job control; None uses the library default. Not accepted by Spark. |
| `random_state` | Seed for supported random estimators/resampling. Spark partition/order changes may still change results. |
| `fold_col` | Spark-only optional existing column of fold IDs in [0,cv), excluded from features and target. Other folds train each model; not expanding-window CV. |

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

Use the [Spark setup](spark.md#example-data) or the [pandas setup](setup.md). From the repository root, run `python examples/feature_selection/method_example.py --backend spark --method sequential_selection`. Use `--backend pandas` for the pandas example.

## How it works

### Spark usage

See [the complete Spark setup and sequential_selection example](spark.md#sequential_selection) for the
distributed algorithm, parameters, result fields, limits, and interpretation.

[Example data](spark.md#example-data)

```python
result = fs.sequential_selection(
    df=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    model="logistic_regression",  # Estimator used to fit the candidate features.
    model_params=None,  # Estimator settings.
    direction="forward",  # Forward adds one feature at a time. Backward starts with all candidates.
    n_features_to_select=1,  # Number of features in the final greedy subset.
    scoring="roc_auc",  # Metric used to evaluate predictions. Here it is area under the ROC curve.
    cv=3,  # Number of cross-validation folds used to score candidate subsets.
    fold_col="fold",  # Column assigning each training row to a validation fold.
    random_state=42,  # Seed for reproducible random choices.
)
print(result.selected_features)
```

### pandas usage

[Example data](setup.md)

**Supervised; Pandas behavior.** Forward selection adds the candidate with the best
cross-validated score to the current subset. Backward selection removes the
candidate whose removal gives the best score. Unlike RFE, it does not require
native importance attributes.

```python
from brainmodelkit.feature_selection.pandas import sequential_selection

result = sequential_selection(
    df_train=df,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    df_oot=None,  # Optional held-out frame for evaluating the selected-feature model.
    df_scoring=None,  # Optional additional frame to score with the final model.
    run_name="feature_selection",  # Label used to identify the training run and its outputs.
    model="logistic_regression",  # Estimator used to fit the candidate features.
    model_params=None,  # Estimator settings.
    direction="forward",  # Forward adds one feature at a time. Backward starts with all candidates.
    n_features_to_select=3,  # Number of features in the final greedy subset.
    scoring="roc_auc",  # Metric used to evaluate predictions. Here it is area under the ROC curve.
    cv=5,  # Number of cross-validation folds used to score candidate subsets.
    n_jobs=None,  # Number of parallel workers. None uses the configured joblib default.
    random_state=42,  # Seed for reproducible random choices.
)
```

At each step, cross-validate each candidate addition or removal and keep the
subset with the highest mean AUC. After reaching the requested count, fit the
estimator on all training rows using the selected columns. The result includes
the selected features and the fitted model.

Useful when performance matters more than a native importance proxy, especially
with a modest feature count. It can require many more fits than RFE and greedy
search need not find the globally best subset. Apply the same leakage and CV split
cautions as RFECV. See [sklearn sequential selection](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html).

## Computational cost and practical limits

Typical relative cost: **very high**, with overlapping tiers across methods.
This is an implementation-based estimate, not a benchmark or a universal ranking.
Spark: distributed CV fits for candidate subsets, then a final fit. Pandas: local CV fits for candidate subsets, then a final fit.
Rows, feature count, cardinality, iterations, folds, and the chosen estimator can
change which method dominates. Use this method for small candidate pools when justified;
do not run every selector as a fixed pipeline.

See the [backend comparison](../selection-costs.md) for an ordered shortlist
and [input contracts](../feature_selection.md) before comparing scores.
