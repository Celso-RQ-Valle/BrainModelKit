# Sequential Feature Selection: sequential_selection

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pandas as fs` for Pandas,
or `from brainmodelkit.feature_selection import pyspark as fs` for Spark.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

**Spark implementation:** read the [Spark algorithm and execution contract](spark.md#sequential_selection).
The explanations below describe shared concepts and the Pandas behavior. Consult
the signatures for backend-specific arguments; they are not interchangeable.

## Signatures

### Pandas

```python
def sequential_selection(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    oot_df=None,
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

### PySpark

```python
def sequential_selection(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
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

The listings show accepted arguments and defaults. Pass optional controls by
keyword; permutation requires a fitted model first and RFE requires run_name first.
`None` placeholders for train/target/features do not make those inputs optional
at call time. Unsupported backend keywords raise TypeError.

## Parameters

| Name | Meaning |
| --- | --- |
| `train_df` | Training DataFrame for the selected backend. Fit selection on development/training rows only. |
| `target_col` | Target column name; exclude it from feature_cols. Required for supervised methods. Spark requires both 0 and 1; Pandas class support is described below. |
| `feature_cols` | Explicit nonempty ordered list of unique existing feature names. RFE alone can infer names starting feat_. Selection outputs preserve this order. |
| `df` | Alias for train_df except permutation, where this is the held-out development DataFrame. Supplying distinct frames under both aliases raises ValueError. |
| `oot_df` | Optional labeled OOT DataFrame for supported importance/wrapper final evaluation; required by historical RFE. It never chooses the feature subset. |
| `df_oot` | Alias for oot_df. Supplying distinct frames under both spellings raises ValueError. |
| `df_scoring` | Optional unlabeled frame to score with the final selected-feature classifier; requires OOT evaluation. RFE scores it only at the final fit. |
| `run_name` | Label used for final training outputs or RFE iteration reports. |
| `model` | Named or compatible native estimator; permutation requires an already fitted model. Model restrictions and return schemas are explained below. |
| `model_params` | Optional dictionary of native estimator parameters: Pandas snake_case, Spark camelCase. Custom estimators are copied/cloned. |
| `direction` | Sequential direction: forward adds one feature each round; backward removes one. |
| `n_features_to_select` | Sequential selection's desired positive feature count, strictly smaller than initial count. |
| `scoring` | Pandas sklearn scorer; Spark accepts roc_auc or average_precision (Spark PR area, not sklearn average precision). |
| `cv` | Pandas fold count or sklearn-compatible splitter/iterable. Spark integer >=2; all validation folds must contain both classes. |
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

Run the [shared example setup](setup.md) before the snippet below, or execute
`python examples/feature_selection/method_example.py --method sequential_selection`
from the repository root for a self-contained Pandas example. Add `--backend spark`
for the Spark implementation. See the method-specific parameter tables above.

## How it works

**Supervised; Pandas behavior.** Forward selection adds the candidate with the best
cross-validated score to the current subset. Backward selection removes the
candidate whose removal gives the best score. Unlike RFE, it does not require
native importance attributes.

```python
from brainmodelkit.feature_selection.pandas import sequential_selection

result = sequential_selection(
    df,
    target_col="target",
    feature_cols=features,
    direction="forward",
    n_features_to_select=3,
    scoring="roc_auc",
    cv=5,
)
```

This uses sklearn `SequentialFeatureSelector`. Set `direction="backward"` for
removal; pass a named/custom model, `model_params`, `cv`, `n_jobs`, and a seed.
Choose a positive count smaller than the input feature count. The result exposes
the selected/rejected features and Boolean `feature_table`; no artificial ranking,
CV history, or fitted model is manufactured because sklearn does not supply them.
Fit the downstream model on the selected columns.

Useful when performance matters more than a native importance proxy, especially
with a modest feature count. It can require many more fits than RFE and greedy
search need not find the globally best subset. Apply the same leakage and CV split
cautions as RFECV. See [sklearn sequential selection](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html).

## Spark usage

See [the complete Spark setup and sequential_selection example](spark.md#sequential_selection) for the
distributed algorithm, parameters, result fields, limits, and interpretation.
