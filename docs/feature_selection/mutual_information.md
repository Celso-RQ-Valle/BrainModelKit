# Mutual Information: mutual_information

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pyspark as fs` for Spark,
or `from brainmodelkit.feature_selection import pandas as fs` for pandas.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

**Spark implementation:** read the [Spark algorithm and execution contract](spark.md#mutual_information).
The explanations below describe shared concepts and the Pandas behavior. Consult
the signatures for backend-specific arguments; they are not interchangeable.

## Signatures

### PySpark

```python
def mutual_information(
    df,
    target_col=None,
    feature_cols=None,
    *,
    threshold=None,
    top_k=None,
    discrete_features=False,
    n_bins=10,
    relative_error=0.001,
    max_categories=1000,
): ...
```

### Pandas

```python
def mutual_information(
    df,
    target_col=None,
    feature_cols=None,
    *,
    threshold=None,
    top_k=None,
    random_state=42,
    discrete_features=False,
    n_neighbors=3,
): ...
```

The listings show accepted arguments and defaults. Pass optional controls by
keyword; permutation requires a fitted model first and RFE requires run_name first.

## Parameters

| Name | Meaning |
| --- | --- |
| `df` | Training DataFrame for the selected backend. Fit selection on development/training rows only. |
| `target_col` | Target column name; exclude it from feature_cols. Required for supervised methods. Spark requires both 0 and 1; Pandas class support is described below. |
| `feature_cols` | Explicit nonempty ordered list of unique existing feature names. RFE alone can infer names starting feat_. Selection outputs preserve this order. |
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

Use the [Spark setup](spark.md#example-data) or the [pandas setup](setup.md). From the repository root, run `python examples/feature_selection/method_example.py --backend spark --method mutual_information`. Use `--backend pandas` for the pandas example.

## How it works

### Spark usage

See [the complete Spark setup and mutual_information example](spark.md#mutual_information) for the
distributed algorithm, parameters, result fields, limits, and interpretation.

[Example data](spark.md#example-data)

```python
result = fs.mutual_information(
    df=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    threshold=None,  # Optional minimum score a feature must reach to qualify.
    top_k=1,  # Optional limit on the number of highest-ranked features retained.
    discrete_features=False,  # Identify discrete inputs. False treats the feature columns as continuous.
    n_bins=10,  # Requested number of bins for continuous values.
    relative_error=0.001,  # Allowed rank error when Spark approximates quantile boundaries.
    max_categories=1000,  # Limit distinct levels per discrete feature to bound counting costs.
)
print(result.selected_features)
```

### pandas usage

[Example data](setup.md)

**Supervised classification; Pandas behavior.** MI measures departure from independence:
`I(X; Y) = sum p(x,y) log(p(x,y)/(p(x)p(y)))`, with the analogous integral for
continuous variables. Scores are in nats. Zero estimated MI means no detected
univariate dependence, not proof of independence.

```python
from brainmodelkit.feature_selection.pandas import mutual_information

result = mutual_information(
    df=df,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    threshold=None,  # Optional minimum score a feature must reach to qualify.
    top_k=5,  # Optional limit on the number of highest-ranked features retained.
    random_state=42,  # Seed for reproducible random choices.
    discrete_features=False,  # Identify discrete inputs. False treats the feature columns as continuous.
    n_neighbors=3,  # Number of neighbors used to estimate dependence for continuous inputs.
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

## Computational cost and practical limits

Typical relative cost: **moderate**, with overlapping tiers across methods.
This is an implementation-based estimate, not a benchmark or a universal ranking.
Spark: quantile binning and joint-count shuffles. Pandas: nearest-neighbor estimation for continuous data.
Rows, feature count, cardinality, iterations, folds, and the chosen estimator can
change which method dominates. Use this method for nonlinear screening when justified;
do not run every selector as a fixed pipeline.

Spark collects bin edges and bounded feature/category summaries, not source rows.
Its binned empirical estimate is not pandas nearest-neighbor MI.

See the [backend comparison](../selection-costs.md) for an ordered shortlist
and [input contracts](../feature_selection.md) before comparing scores.
