# Mutual Information: mutual_information

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pandas as fs` for Pandas,
or `from brainmodelkit.feature_selection import pyspark as fs` for Spark.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

**Spark implementation:** read the [Spark algorithm and execution contract](spark.md#mutual_information).
The explanations below describe shared concepts and the Pandas behavior. Consult
the signatures for backend-specific arguments; they are not interchangeable.

## Signatures

### Pandas

```python
def mutual_information(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    threshold=None,
    top_k=None,
    random_state=42,
    discrete_features=False,
    n_neighbors=3,
): ...
```

### PySpark

```python
def mutual_information(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    threshold=None,
    top_k=None,
    discrete_features=False,
    n_bins=10,
    relative_error=0.001,
    max_categories=1000,
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
| `threshold` | Inclusive score cutoff; None disables it. For correlation, reject at absolute correlation >= threshold in [0,1]. |
| `top_k` | Optional positive integer rank limit no larger than candidate count. Intersects other gates; ties favor input order. |
| `random_state` | Seed for supported random estimators/resampling. Spark partition/order changes may still change results. |
| `discrete_features` | Boolean or per-feature Boolean mask. Marks numeric encoded levels as discrete; continuous estimation differs across backends. |
| `n_neighbors` | Pandas continuous MI estimator neighborhood size, integer >=1; affects bias/variance. |
| `n_bins` | Number of requested numeric bins, integer >=2. Duplicate boundaries can reduce effective bins. |
| `relative_error` | Spark approximate quantile rank error in [0,1]; 0 requests exact quantiles at higher cost. |
| `max_categories` | Spark discrete cardinality guard, integer >=2. Encode/bin high-cardinality data explicitly. |

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
`python examples/feature_selection/method_example.py --method mutual_information`
from the repository root for a self-contained Pandas example. Add `--backend spark`
for the Spark implementation. See the method-specific parameter tables above.

## How it works

**Supervised classification; Pandas behavior.** MI measures departure from independence:
`I(X; Y) = sum p(x,y) log(p(x,y)/(p(x)p(y)))`, with the analogous integral for
continuous variables. Scores are in nats. Zero estimated MI means no detected
univariate dependence, not proof of independence.

```python
from brainmodelkit.feature_selection.pandas import mutual_information

result = mutual_information(
    df, target_col="target", feature_cols=features, top_k=5, random_state=42
)
```

The implementation uses sklearn's estimator. `discrete_features=False` assumes
continuous features; pass `True` or a per-feature Boolean mask for encoded
discrete values. `n_neighbors=3` controls continuous estimation. A fixed seed
controls the noise used to break repeated-value ties. Outputs are `feature`,
`mutual_information`, `ranking`, and `selected`; `threshold=None` and `top_k=None`
mean report/select all defined estimates.

Useful for nonlinear univariate screening. It needs adequate sample sizes and
can be noisy for sparse categories; arbitrary ordinal encoding of nominal
categories as continuous values is inappropriate. It does not account for
redundancy or interactions. Spark implements binned MI; read the [Spark estimator contract](spark.md#mutual_information). See [sklearn MI](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html).

## Spark usage

See [the complete Spark setup and mutual_information example](spark.md#mutual_information) for the
distributed algorithm, parameters, result fields, limits, and interpretation.
