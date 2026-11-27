# RFECV: rfecv

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pyspark as fs` for Spark,
or `from brainmodelkit.feature_selection import pandas as fs` for pandas.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

**Spark implementation:** read the [Spark algorithm and execution contract](spark.md#rfecv).
The explanations below describe shared concepts and the Pandas behavior. Consult
the signatures for backend-specific arguments; they are not interchangeable.

## Signatures

### PySpark

```python
def rfecv(
    df,
    target_col=None,
    feature_cols=None,
    *,
    model="logistic_regression",
    model_params=None,
    scoring="roc_auc",
    cv=5,
    fold_col=None,
    step=1,
    min_features_to_select=1,
    random_state=42,
): ...
```

### Pandas

```python
def rfecv(
    df_train,
    target_col=None,
    feature_cols=None,
    *,
    df_oot=None,
    df_scoring=None,
    run_name="feature_selection",
    model="logistic_regression",
    model_params=None,
    scoring="roc_auc",
    cv=5,
    step=1,
    min_features_to_select=1,
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
| `scoring` | Spark accepts roc_auc or average_precision (Spark PR area). Pandas accepts a sklearn scorer. |
| `cv` | Spark takes an integer >=2 and requires both classes in every validation fold. Pandas accepts a fold count or sklearn-compatible splitter/iterable. |
| `step` | Positive integer number of weakest features removed per RFE/RFECV step; final removal is clipped to the minimum. |
| `min_features_to_select` | RFECV's smallest tested subset, positive integer no greater than initial feature count. |
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

Use the [Spark setup](spark.md#example-data) or the [pandas setup](setup.md). From the repository root, run `python examples/feature_selection/method_example.py --backend spark --method rfecv`. Use `--backend pandas` for the pandas example.

## How it works

### Spark usage

See [the complete Spark setup and rfecv example](spark.md#rfecv) for the
distributed algorithm, parameters, result fields, limits, and interpretation.

[Example data](spark.md#example-data)

```python
result = fs.rfecv(
    df=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    model="logistic_regression",  # Estimator used to fit the candidate features.
    model_params=None,  # Estimator settings.
    scoring="roc_auc",  # Metric used to evaluate predictions. Here it is area under the ROC curve.
    cv=3,  # Number of cross-validation folds used to score candidate subsets.
    fold_col="fold",  # Column assigning each training row to a validation fold.
    step=1,  # Number of features removed in each elimination round.
    min_features_to_select=1,  # Smallest subset size considered during elimination.
    random_state=42,  # Seed for reproducible random choices.
)
print(result.selected_features)
```

### pandas usage

[Example data](setup.md)

**Supervised; Pandas behavior.** RFECV means RFE plus cross-validation used to choose
the feature count. It is a separate sklearn-based selector, not a change to
BrainModelKit's OOT RFE or its persistence contract.

```python
from brainmodelkit.feature_selection.pandas import rfecv

result = rfecv(
    df_train=df,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    df_oot=None,  # Optional held-out frame for evaluating the selected-feature model.
    df_scoring=None,  # Optional additional frame to score with the final model.
    run_name="feature_selection",  # Label used to identify the training run and its outputs.
    model="logistic_regression",  # Estimator used to fit the candidate features.
    model_params=None,  # Estimator settings.
    scoring="roc_auc",  # Metric used to evaluate predictions. Here it is area under the ROC curve.
    cv=5,  # Number of cross-validation folds used to score candidate subsets.
    step=1,  # Number of features removed in each elimination round.
    min_features_to_select=2,  # Smallest subset size considered during elimination.
    n_jobs=None,  # Number of parallel workers. None uses the configured joblib default.
    random_state=42,  # Seed for reproducible random choices.
)
print(result.optimal_feature_count, result.ranking)
print(result.cv_results)
```

Each elimination step fits an importance-producing estimator on all training rows
and cross-validates the current subset. `cv_results` records the feature count,
mean AUC, and AUC standard deviation for each visited subset. Rank 1 marks
selected features. `model` is the estimator refitted on the selected columns.

RFECV is useful when feature count should be chosen by a CV objective. It is
expensive, and ordinary stratified folds are inappropriate for temporal dependence.
CV used for selection is not an independent final performance estimate. Existing
RFE's OOT metric path is neither calculated nor persisted by this method.
See [sklearn RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html).

## Computational cost and practical limits

Typical relative cost: **very high**, with overlapping tiers across methods.
This is an implementation-based estimate, not a benchmark or a universal ranking.
Spark: full-data elimination, CV fits per subset, then elimination again to the chosen size. Pandas: full-data elimination, CV fits per subset, then one refit on the chosen subset.
Rows, feature count, cardinality, iterations, folds, and the chosen estimator can
change which method dominates. Use this method for budgeted subset search when justified;
do not run every selector as a fixed pipeline.

See the [backend comparison](../selection-costs.md) for an ordered shortlist
and [input contracts](../feature_selection.md) before comparing scores.
