# Spark selection: algorithms and execution contracts

[Start in the README](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)
or return to the [method index](../feature_selection.md#api-quick-reference).
All sixteen methods are exported from `brainmodelkit.feature_selection.pyspark`.
The seven methods below complete the Spark namespace. Their full signatures are
listed on the individual method pages linked from the index.

Install `pip install "BrainModelKit[pyspark]"` and configure Java and a classic
Spark session. NumPy handles feature-sized diagnostic arrays; SciPy evaluates
scalar F and binomial distribution tails. Neither Pandas nor sklearn is used.
Permutation and Boruta use distributed Python RDD indexing and require classic
Spark, not Spark Connect. No selector collects the input dataset to the driver.

Features must be unique finite numeric columns, explicitly encoded and imputed.
Every supervised Spark selector requires both binary target classes, 0 and 1.
Select on development data and keep the final test set untouched. All new methods
return `SelectionResult`; `selected_features` always preserves the input order.
An empty selection is valid. Check it before training a downstream classifier.

The new wrappers accept selection data only; they do not take `oot_df`,
`df_scoring`, or persistence arguments. Train/evaluate the resulting list using
`brainmodelkit.training.pyspark.train_model`. Existing importance/L1 methods still
support optional final OOT evaluation, and existing RFE writes iteration reports.

## Example data

Run this setup once before the examples on this page:

```python
from pyspark.sql import SparkSession, functions as F
from brainmodelkit.feature_selection import pyspark as fs

spark = (
    SparkSession.builder.master("local[2]").appName("feature-selection").getOrCreate()
)
data = (
    spark.range(320)
    .withColumn("target", (F.col("id") % 2).cast("int"))
    .withColumn("signal", F.col("target").cast("double"))
    .withColumn("noise", F.rand(42))
    .withColumn("constant", F.lit(1.0))
    .withColumn("fold", (F.floor(F.col("id") / 2) % 3).cast("int"))
)
train = data.filter("id < 240")
validation = data.filter("id >= 240")
features = ["signal", "noise", "constant"]
```

This deliberately easy synthetic dataset tests mechanics; the perfect signal is
not a realistic benchmark. Real fold assignments must respect entity and time
boundaries. The examples below are independent, not a recommended chain of every
expensive method. Stop your caller-owned session after finishing.

## mutual_information

Spark estimates `sum(n_xy/N * log(n_xy*N/(n_x*n_y)))` from distributed joint
counts. `discrete_features=True` uses exact numeric levels; `False` bins each
feature at training quantiles. A Boolean mask chooses per-column behavior.
Equality at a cut belongs to the lower bin. Duplicate edges collapse; a constant
has MI zero. Counts are unsmoothed, and negative roundoff is clipped to zero.

`n_bins=10` sets quantile resolution; `relative_error=0.001` bounds approximate
quantile rank error (`0` requests exact quantiles). `max_categories=1000` bounds
each discrete feature's cardinality. `threshold=None` selects every defined score;
`top_k=None` imposes no rank cap. A supplied threshold is inclusive and a supplied
top_k intersects it. Ties favor earlier features. Inspect
`metadata['bin_edges']`, `metadata['discrete_features']`, and each feature's
`mutual_information` and `ranking`.

```python
mi = fs.mutual_information(
    train, "target", features, discrete_features=[True, False, True], top_k=1
)
print(mi.selected_features)  # ['signal']
```

This is empirical MI of discretized variables, not sklearn's continuous nearest
neighbor estimator. It loses within-bin information and has finite-sample upward
bias, especially with many categories. It has no p-value or automatic significance
test. Select binning on training data; never compare independently re-binned OOT
MI as if it were a fitted transformation. There is no Spark `n_neighbors` or
`random_state` parameter for this deterministic counting estimator.

## anova

For each binary class c, aggregate its size `n_c`, mean `m_c`, and sample
variance `s_c²`. With global mean m, calculate
`SS_between = sum(n_c*(m_c-m)²)` and
`SS_within = sum((n_c-1)*s_c²)`. The statistic is
`F = SS_between / (SS_within/(N-2))`; the upper F(1,N-2) tail gives the p-value.
Only two class summaries per feature are collected. Positive between-class
variation with zero within-class variation returns infinity and p=0. A global
constant has undefined F and is rejected. N must exceed 2.

```python
test = fs.anova(train, "target", features, max_p_value=0.05)
print(test.feature_table)
```

The null hypothesis is equal class means. Classical calibration assumes
independent observations, normal within-class errors, and equal variances.
`max_p_value=0.05` is an inclusive, unadjusted gate; set it to `None` to disable
it. `top_k` optionally intersects the gate, ranking decreasing F. Tests across
many features need a separately chosen multiplicity policy. A low p-value is
not an effect-size measure and a large p-value does not establish independence.
See also [Spark's univariate test documentation](https://spark.apache.org/docs/3.5.7/ml-features.html#univariatefeatureselector).

## permutation_importance_selection

The caller supplies a fitted native Spark probabilistic classifier and held-out
development data. Assemble features in exactly the model's original training
order. Compute a baseline score, independently shuffle one column, then compute
`baseline - shuffled_score`. Repeat `n_repeats=5` times per column and return
`importance_mean`, population `importance_std`, `ranking`, and `selected`.
Each shuffle preserves the column's multiset including duplicates.

```python
fitted = fs.feature_importance_selection(
    train, "target", features, model="decision_tree"
).model
permutation = fs.permutation_importance_selection(
    fitted,
    validation,
    "target",
    features,
    n_repeats=3,
    top_k=1,
)
print(permutation.feature_table)
```

`scoring='roc_auc'` uses Spark's ROC area with `numBins=0`;
`'average_precision'` maps to Spark `areaUnderPR`, which is PR area and is **not**
sklearn's average precision integration convention. No other scorer is supported.
`random_state=42` seeds shuffles. `threshold` and `top_k` follow MI's rules;
negative importance is permitted and does not imply a protective causal effect.
Correlated predictors can mask one another. No model is fitted or mutated here.

Indexing uses distributed `zipWithIndex`, random sorting, and joins. An owned
cache stabilizes input row IDs and is released afterward. Each repeat
costs a distributed sort, join, and prediction; avoid running on hundreds of raw
features. Classic Spark Python workers are required. Seeds reproduce operations
on stable input order/partitions, not across arbitrary repartitioning.

## rfecv

`cv=5` defines complementary train/validation folds. Supply `fold_col` with IDs
0 through cv-1 for reproducible entity-aware splits; otherwise a seeded random
assignment is materialized once. Every validation fold must contain both classes.
Fold membership is never a candidate feature. This API does not implement expanding
time windows: all other folds are used for training, including later folds.

Inside each training fold, repeatedly fit, remove up to `step=1` smallest absolute
importances, and stop at `min_features_to_select=1`. Score each subset on that
fold's validation data. Select the visited count with highest mean score; exact
ties choose the smaller count. Finally repeat elimination on all training data
to that count. Crucially, validation labels never influence a fold's elimination
ordering. Counts skipped by a larger step are not evaluated.

```python
cv_result = fs.rfecv(
    train,
    "target",
    features,
    model="decision_tree",
    cv=3,
    fold_col="fold",
    min_features_to_select=1,
)
print(cv_result.selected_features)
print(cv_result.cv_results)
```

`model='logistic_regression'`, `model_params=None`, and `random_state=42` configure
the native estimator. Supported model names are those of feature importance.
Models must expose finite importance or coefficients. `scoring` has the same
two options as permutation. There is no local `n_jobs` option: Spark controls job
execution. `cv_results` contains ascending `n_features`, `mean_test_score`,
population `std_test_score`, and per-count `split_scores`. `history` records the
full-data path; rank 1 is selected, later elimination rounds have worse ranks.
`model` is fitted on `metadata['model_feature_cols']` only.

The chosen CV maximum is selection-biased; use nested CV or untouched final
evaluation to estimate generalization. The cost is roughly cv times the number
of visited counts, plus a full-data elimination path. Preprocessing must be fitted
within folds to avoid leakage; the selector does not create preprocessing pipelines.

## sequential_selection

Use the same fold/scoring/model controls as RFECV. `direction='forward'` starts
empty and tests adding each remaining feature; `'backward'` starts full and tests
each removal. Fit every candidate separately in each training fold, choose the
highest mean validation score, and repeat to `n_features_to_select=1`. Exact ties
favor the candidate encountered first in input order. The requested count must
be at least 1 and strictly smaller than the candidate count.

```python
subset = fs.sequential_selection(
    train,
    "target",
    features,
    model="decision_tree",
    direction="forward",
    n_features_to_select=2,
    cv=3,
    fold_col="fold",
)
print(subset.history)
```

The final native model is refitted on all training rows and selected columns.
`history` records each winner and all candidate mean scores. This greedy search
does not guarantee the globally best subset and can miss pure interactions in
forward mode. The implementation uses the shared importance-capable native model
factory. Repeated candidate fitting can require O(cv*p²) model fits.

## stability_selection

For each iteration, Spark samples with replacement using Poisson row multiplicities
with mean `sample_fraction=0.8`, fits either `method='feature_importance'` or
`'l1'`, and records selected features. `n_iterations=20` controls fits per group.
Select a feature when `selection_count/n_fits >= min_frequency=0.8`.
Return frequencies, counts, mean absolute importance, population standard
deviation, and fit/group history. No final classifier is returned.

```python
stable = fs.stability_selection(
    train,
    "target",
    features,
    n_iterations=5,
    selector_kwargs={"model": "decision_tree", "top_k": 1},
)
print(stable.feature_table)
```

`group_by=None` uses the whole frame. A column name or list splits fits across
observed groups including null keys; `max_groups=100` limits collected group keys.
Each group contributes equally many fits regardless of size. Every group and
bootstrap must contain both classes; a sample losing a class raises an error
instead of being silently retried. Small or rare-class groups can therefore fail.
Spark sampling is not Pandas's fixed-size stratified bootstrap.

`selector_kwargs` configures the base selector (Spark `reg_param` for L1).
Importance selection requires a threshold or top_k; an empty options dictionary
defaults top_k to max(1,p//2). Pass resampling `random_state` at the outer level.
Group order and partition layout can affect seeds. This is a repeatability
diagnostic with no formal false-discovery guarantee. Each owned bootstrap
cache is released, including after errors.

## boruta

See the [Boruta algorithm and inference guide](boruta.md) for shadows, hit tests,
hypotheses, correction procedures, output interpretation, and a complete example.

```python
all_relevant = fs.boruta(
    train,
    "target",
    features,
    n_estimators=30,
    model_params={"maxDepth": 4},
    max_iter=30,
)
print(all_relevant.feature_table)
```

This is the most expensive new method: every iteration permutes at least five
shadow columns, joins them to the real data, and fits a forest. Start with a small
candidate set and inspect tentative decisions before increasing the iteration
budget. Internal caches require executor storage. Functions release their own
caches without unpersisting caller-owned cached inputs; a unique internal marker
prevents identical-plan cache ownership collisions.
