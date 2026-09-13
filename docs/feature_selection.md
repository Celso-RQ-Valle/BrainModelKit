# Feature Selection Guide

## Overview

BrainModelKit offers independent selectors that compose through
`result.selected_features`. Import from `brainmodelkit.feature_selection.pandas`
for Pandas DataFrames or `brainmodelkit.feature_selection.pyspark` for Spark
DataFrames. The backends do not delegate to one another. There is no automatic
pipeline engine and no implicit category encoding or imputation.

Install `pip install "BrainModelKit[pandas]"` or
`pip install "BrainModelKit[pyspark]"`. Spark requires a caller-configured Spark
session and Java installation. Boruta additionally needs
`pip install "BrainModelKit[pandas,boruta]"`.

Core installation remains dependency-free. The Spark extra includes NumPy for
feature-sized arrays and SciPy for scalar statistical tails, without adding Pandas
or sklearn.
See [Spark's dependency requirements](https://spark.apache.org/docs/3.5.7/api/python/getting_started/install.html).

Except for the existing RFE, selectors return a lightweight `SelectionResult`:

| Attribute | Meaning |
| --- | --- |
| `selected_features` / `rejected_features` | Feature names in original input order |
| `feature_table` | List of dictionaries, one per input feature, including `selected` |
| `method` / `metadata` | Method identifier and method-specific configuration |
| `ranking` | Feature-to-rank mapping for methods that produce ranks |
| `grouped_table` | Completeness diagnostics: Pandas or distributed Spark DataFrame |
| `correlation_pairs` | List of high-correlation feature pairs and coefficients |
| `bin_details` | WoE diagnostics: Pandas or distributed Spark DataFrame |
| `cv_results` | RFECV CV scores/counts; backend-specific dictionaries |
| `history` | Stability fit/group diagnostics; RFE has its own history contract |
| `model` | Fitted estimator when the method returns one |

For Pandas presentation, `pd.DataFrame(result.feature_table)` is convenient.
Spark never converts the source dataset to Pandas: only bounded model/statistic
summaries reach Python. New selectors do not persist files or start MLflow runs. Spark search methods
own temporary executor caches and release them afterward.
Existing RFE retains its original `RFEResult` and persistence behavior.

All methods require an explicit, nonempty `feature_cols` list, except historical
RFE, which can infer `feat_` columns. Names must exist, be unique, and exclude the
target. Empty datasets are rejected. Quality filters support missing observations;
variance ignores missing values. IV has a separate missing bin. Other new methods
require finite numeric features with no missing values: encode/impute explicitly.
Numeric infinities are rejected by statistical and variance methods.

Pandas supervised statistics and importance support classification targets with
at least two classes. IV and L1 require binary 0/1 targets. All Spark supervised
selectors and existing training/RFE require both 0 and 1. Targets cannot be missing.

For ranked selectors, higher scores are better; ties favor earlier input features.
`threshold` is inclusive and `top_k` limits the highest ranks. When both are given,
both conditions must hold. Without either, all defined scores are selected: an
analysis-only call does not invent a cutoff. Undefined scores are rejected.
Statistical tests instead use `max_p_value=0.05` by default. Their `top_k` constraint
is intersected with the p-value gate, not used to refill rejected positions.

## API quick reference

Use the backend-specific namespace that matches your DataFrame type. Every link
below opens the detailed section with parameter guidance and a runnable example.

| Function | Pandas | PySpark |
| --- | --- | --- |
| `completeness` | [guide](feature_selection/completeness.md) | [guide](feature_selection/completeness.md) |
| `variance_filter` | [guide](feature_selection/variance_filter.md) | [guide](feature_selection/variance_filter.md) |
| `cardinality` | [guide](feature_selection/cardinality.md) | [guide](feature_selection/cardinality.md) |
| `correlation_filter` | [guide](feature_selection/correlation_filter.md) | [guide](feature_selection/correlation_filter.md) |
| `mutual_information` | [guide](feature_selection/mutual_information.md) | [guide](feature_selection/mutual_information.md) |
| `chi_square` | [guide](feature_selection/chi_square.md) | [guide](feature_selection/chi_square.md) |
| `anova` | [guide](feature_selection/anova.md) | [guide](feature_selection/anova.md) |
| `information_value` | [guide](feature_selection/information_value.md) | [guide](feature_selection/information_value.md) |
| `feature_importance_selection` | [guide](feature_selection/feature_importance_selection.md) | [guide](feature_selection/feature_importance_selection.md) |
| `l1_selection` | [guide](feature_selection/l1_selection.md) | [guide](feature_selection/l1_selection.md) |
| `permutation_importance_selection` | [guide](feature_selection/permutation_importance_selection.md) | [guide](feature_selection/permutation_importance_selection.md) |
| `rfe` | [guide](feature_selection/rfe.md) | [guide](feature_selection/rfe.md) |
| `rfecv` | [guide](feature_selection/rfecv.md) | [guide](feature_selection/rfecv.md) |
| `sequential_selection` | [guide](feature_selection/sequential_selection.md) | [guide](feature_selection/sequential_selection.md) |
| `stability_selection` | [guide](feature_selection/stability_selection.md) | [guide](feature_selection/stability_selection.md) |
| `boruta` | [guide](feature_selection/boruta.md) | [guide](feature_selection/boruta.md#spark-example-and-parameters) |

Basic usage follows the same pattern for all composable selectors:

```python
from brainmodelkit.feature_selection.pandas import completeness

result = completeness(df, feature_cols=["income", "age"])
selected = result.selected_features
diagnostics = result.feature_table
```

The public modules also expose this guide from their module and function
docstrings. RFE is retained as the training-integrated API and returns its
backward-compatible `RFEResult`; all other selectors return `SelectionResult`.

## Evaluation outputs

Importance and L1 in both backends, and Pandas RFECV, sequential, stability, and
Boruta accept `oot_df` (alias `df_oot`). Selection uses training data only. With a
nonempty selection the final classifier is refitted on selected training columns,
evaluated on OOT, and optionally scores `df_scoring`. Read `training_result`,
`metrics`, and predictions from the result; no persistence is requested. If no
feature survives, evaluation is skipped and `metadata['evaluation_status']` is
`no_selected_features`. The original selection estimator is kept in
`selection_model` when present. The new Spark wrappers accept selection data only;
pass their selected list to `training.pyspark.train_model` for final evaluation.

## How to choose a feature-selection method

Start with availability and constant-feature checks. Inspect cardinality before
encoding and correlation after numerical preprocessing. Use supervised filters
for inexpensive screening, then model-based or wrapper methods when predictive
interactions matter. Use stability diagnostics when results must hold across
resamples or time periods. Boruta answers an all-relevant question rather than
targeting a compact subset.

Selection is part of model fitting. Fit supervised filters, bin edges, encoders,
and imputation on training data only. Do not screen the full dataset before
cross-validation and then interpret CV scores as unbiased estimates. Hold out a
final evaluation set; use appropriate temporal or grouped splits when rows are
dependent. A univariate rejection does not prove a feature is useless in an
interaction. P-values are unadjusted; screening many features increases false
discoveries. None of these methods establishes causality.

## Quality filters

### Completeness

**Unsupervised; Pandas and native Spark.** Completeness measures data availability,
not predictive power:

```text
completeness = non-null observations / total observations
missing_rate = 1 - completeness
```

Pandas missing values and Spark floating NaN count as missing, as do Spark SQL
nulls. Empty strings and sentinel values such as `-999` are not automatically
missing; normalize them first if that is their business meaning. Selection uses
`completeness >= min_completeness` (default 0.70). An all-missing feature therefore
passes only at an explicitly chosen zero threshold.

```python
from brainmodelkit.feature_selection.pandas import completeness

result = completeness(
    df, feature_cols=features, min_completeness=0.70, group_by="reference_date"
)
print(result.selected_features)
print(result.grouped_table)
```

`feature_table` contains `feature`, `completeness`, `missing_rate`, and `selected`.
Use `group_by=["reference_date", "segment"]` for multi-column analysis. Grouped
diagnostics include missing group keys. A trajectory such as Jan 99%, Feb 98%,
Mar 76%, Apr 42% can expose collection failures hidden by a high global average.
Group rows have their own threshold flags, but by default only global completeness
chooses features. Set `require_all_groups=True` to require every observed group
to pass as well. This option requires `group_by`; it can be sensitive to tiny
groups. Group column names cannot collide with diagnostic output names.

Use completeness early for operational reliability. Do not use it as a substitute
for predictive evaluation: missingness can itself carry information, and a rare
but valuable field may merit a dedicated model. Spark computes means of native
non-null indicators; group tables stay distributed.

### Variance

**Unsupervised; Pandas and native Spark.** Population variance is
`sum((x - mean(x))**2) / n`, calculated over non-missing observations (`ddof=0`).
Features pass only when `variance > min_variance`; the default zero removes
constants. A single observed value has zero variance; all-missing features have
undefined variance and are rejected.

```python
from brainmodelkit.feature_selection.pandas import variance_filter

result = variance_filter(df, feature_cols=features, min_variance=0.0)
```

Outputs are `feature`, `variance`, and `selected`. Numeric features are required.
This is useful for eliminating constants before expensive fitting or correlation.
Avoid a common nonzero cutoff across incomparable units: variance depends on
scale, and low-variance rare indicators can still be predictive. Standardization
before this filter makes a nonzero variance threshold largely uninformative.
Spark uses native `var_pop` aggregation.

### Cardinality

**Unsupervised; Pandas and native Spark.** Cardinality counts distinct non-missing
values. `unique_ratio = unique_count / total_rows`, including missing rows in the
denominator. Defaults are `min_unique=2`, `max_unique=None`, and
`max_unique_ratio=None`; specified bounds are inclusive.

```python
from brainmodelkit.feature_selection.pandas import cardinality

result = cardinality(
    df, feature_cols=features, min_unique=2, max_unique=100, max_unique_ratio=0.50
)
```

Outputs are `feature`, `unique_count`, `unique_ratio`, and `selected`. Use it to
identify constants, identifier-like fields, or costly categorical encodings.
Do not automatically discard continuous numeric features because their unique
ratio is high. Counts are exact, not approximate; Spark distinct aggregation can
require substantial shuffle work. Consider the dataset's sample size and use
completeness separately rather than treating the ratio as coverage.

## Statistical filters

### Correlation

**Unsupervised; Pandas and native Spark.** Pearson correlation is normalized
covariance; Spearman correlation is Pearson correlation of ranks and captures
monotonic rather than only linear relationships. `method` accepts `"pearson"`
or `"spearman"`; both use absolute correlation for filtering.

```python
from brainmodelkit.feature_selection.pandas import correlation_filter

result = correlation_filter(
    df, feature_cols=features, threshold=0.90, method="spearman"
)
print(result.correlation_pairs)
```

The rule is deterministic: traverse features in input order and reject a feature
if its absolute correlation is at least the threshold with any **previously
retained** feature. Earlier retained features win. A rejected feature cannot
subsequently reject another feature. All high-correlation pairs are reported,
including pairs involving rejected features. Diagnostics include `rejected_by`.
Constant-feature correlations are undefined and do not trigger rejection; apply
variance filtering first. No target is used in deciding which representative wins.

Use this to reduce redundancy and coefficient instability, not to measure target
association. It misses non-monotonic dependencies and can discard variables that
are useful jointly. At least two complete finite numeric rows are required.
Spark uses native `Correlation.corr`; its matrix needs O(p²) driver memory.
`max_features=256` prevents an unexpectedly large matrix; prefilter or explicitly
raise that budget. Spearman requires distributed ranking and is more expensive.
See the [Spark correlation API](https://spark.apache.org/docs/3.5.7/api/python/reference/api/pyspark.ml.stat.Correlation.html).

### Mutual Information

**Supervised classification; both backends with different estimators.** MI measures departure from independence:
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
redundancy or interactions. Spark uses distributed binned counts; see the
[Spark MI contract](feature_selection/spark.md#mutual_information).
See [sklearn MI](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html).

### Chi-Square

**Supervised; Pandas and native Spark, with different input semantics.** The
statistic sums `(observed - expected)**2 / expected` under independence. Small
p-values indicate evidence against independence, subject to adequate expected
cell counts and independent observations.

```python
from brainmodelkit.feature_selection.pandas import chi_square

result = chi_square(
    counts_df, target_col="target", feature_cols=count_features, max_p_value=0.05
)
```

Pandas uses sklearn `chi2` on **nonnegative integer counts or indicators**. Spark
uses native `ChiSquareTest` on **nonnegative integer category codes**, treating
each distinct value as a categorical level. These statistics are not numerically
interchangeable. Both reject negative or fractional features. Discretize or encode
explicitly; never pass arbitrary continuous values and interpret them as counts.
Spark's `max_categories=1000` guards categorical contingency-table size; Spark
targets are binary 0/1. Pandas supports multiclass classification.

Outputs include `feature`, `statistic`, `p_value`, `ranking`, and `selected`.
Selection requires `p_value <= max_p_value` (default 0.05), optionally intersected
with `top_k`; use `max_p_value=None` to disable that gate. Constant/undefined
features are rejected. Useful for count, indicator, or explicitly categorical
screening, but sparse expected counts and many simultaneous tests weaken p-value
interpretation. See [sklearn chi2](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html)
and [Spark ChiSquareTest](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/stat/ChiSquareTest.html).

### ANOVA / F-test

**Supervised classification; Pandas and Spark (`anova`).** One-way ANOVA compares
between-class and within-class variation: `F = MS_between / MS_within`. It tests
whether class means differ, not arbitrary dependence. The classical p-value
assumes independent observations, approximately normal within-class errors, and
comparable class variances.

```python
from brainmodelkit.feature_selection.pandas import anova

result = anova(df, target_col="target", feature_cols=features, max_p_value=0.05)
```

This wraps sklearn `f_classif`. Outputs are `feature`, `f_statistic`, `p_value`,
`ranking`, and `selected`; `top_k` and `max_p_value` work as for Chi-Square.
There must be more observations than classes. Undefined scores are rejected;
a perfect class separation can legitimately produce an infinite F-statistic
with a zero p-value. Useful for inexpensive numerical screening. Avoid relying
on its p-values under severe heteroskedasticity or dependent observations, and
do not expect it to find symmetric/nonlinear relationships with equal means.
Spark computes equivalent F statistics from distributed class summaries;
see [Spark ANOVA](feature_selection/spark.md#anova).

### Information Value

**Supervised binary classification; Pandas and native Spark.** IV compares the
event and non-event distributions across a feature's bins. Event means target 1.
For observed bin b, additive smoothing alpha gives:

```text
de[b] = (events[b] + alpha) / (total_events + alpha * observed_bin_count)
dn[b] = (non_events[b] + alpha) / (total_non_events + alpha * observed_bin_count)
WoE[b] = log(de[b] / dn[b])
IV = sum((de[b] - dn[b]) * WoE[b])
```

```python
from brainmodelkit.feature_selection.pandas import information_value

result = information_value(
    df,
    target_col="target",
    feature_cols=features,
    n_bins=10,
    min_iv=0.02,
    smoothing=0.5,
    binning="quantile",
)
print(result.bin_details)
```

Numeric variables use `binning="quantile"` (default) or `"uniform"` equal-width
bins. `n_bins >= 2`; duplicate boundaries can reduce effective bins. Categorical
variables use observed categories directly. Missing values have a separate bin,
not a reserved string that could collide with a real category. Both classes must
exist globally. `smoothing > 0` prevents infinite WoE for bins with only one class;
empty bins are excluded. An all-missing or constant feature has zero IV.

`feature_table` contains `feature`, `iv`, `ranking`, and `selected`. `min_iv=None`
reports/selects every defined IV; otherwise the cutoff is inclusive. `bin_details`
contains event/non-event counts, both smoothed distributions, `woe`,
`iv_component`, `bin`, and `is_missing`. Pandas also provides category labels.
Numeric cut points are in `metadata["bin_edges"]`.

Both backends put equality at a cut point in the lower bin.
Spark uses approximate quantiles (`relative_error=0.001`) and native SQL bin comparisons
with upper-inclusive intervals; Pandas uses exact sample quantiles. Their bins
and IV can differ at ties or approximate boundaries. Use the diagnostics rather
than assuming bitwise backend equality. Spark collects only bin edges and a few
scalar totals; all categorical counts and WoE rows stay distributed.

IV is useful for transparent binary univariate screening and bin-level risk
diagnostics. High IV can reveal leakage, rare-category overfit, or excessive
binning, not necessarily a good feature. Compare on development data and use
independent validation. This selector does not return a reusable WoE transformer;
do not independently re-bin OOT data and treat it as a fitted transformation.
It is unsuitable for multiclass targets without a separately designed extension.

## Model-based methods

### Feature Importance

**Supervised; Pandas and native Spark.** `feature_importance_selection` fits a
fresh estimator and ranks its native tree importance or absolute linear
coefficient. The signed binary coefficient remains in `importance`; coefficient
arrays are also in `coefficients`. For Pandas multiclass models, the coefficient
L2 norm across classes determines importance and all signed coefficients remain
in diagnostics. Training/RFE's importance record convention is reused.

```python
from brainmodelkit.feature_selection.pandas import feature_importance_selection

result = feature_importance_selection(
    df,
    target_col="target",
    feature_cols=features,
    model="random_forest",
    model_params={"n_estimators": 100},
    top_k=5,
)
```

Supported names are `logistic_regression`, `decision_tree`, `random_forest`,
`gradient_boosting`, and `lightgbm`, plus compatible estimator instances.
Pandas estimators are cloned; Spark estimators are copied. Use native parameter
names in `model_params` (e.g. Spark `numTrees`). Named stochastic estimators use
`random_state=42`/Spark seed by default; custom estimator randomness remains the
caller's responsibility. Pandas LightGBM needs its optional extra. Spark LightGBM
requires SynapseML and matching JVM packages configured outside BrainModelKit.

Outputs include `feature`, `importance`, `ranking`, and `selected`, plus `model`.
Without oot_df, this is selection only. Supplying oot_df refits the selected
features for final OOT evaluation; see [evaluation outputs](#evaluation-outputs). Spark returns the fitted native
classifier expecting an assembled `features` vector, not a raw-column pipeline.
Models must expose one finite native importance per feature; arbitrary pipelines
without a compatible importance attribute are not implicitly unwrapped.

Use this when importance should reflect a model family. Tree impurity importance
can favor high-cardinality features, and correlated predictors can share or mask
importance. Coefficient magnitudes depend on scaling. An importance is not a
causal effect and does not guarantee held-out benefit. Use permutation or temporal
evaluation where those limitations matter.

### L1 Selection

**Supervised binary classification; Pandas and native Spark.** L1 logistic
regression penalizes `sum(abs(coefficient))`, encouraging exact zeros. Selection
requires `abs(coefficient) > coefficient_tolerance` (default `1e-8`).

```python
from brainmodelkit.feature_selection.pandas import l1_selection

result = l1_selection(df, target_col="target", feature_cols=features, C=1.0)
```

Pandas uses L1/liblinear with `C > 0`, an inverse regularization strength, and
`max_iter=1000`. Spark uses native logistic regression with `elasticNetParam=1`,
`reg_param=0.1`, and `max_iter=100`. Spark's regularization parameter is **not**
an alias for sklearn C. Diagnostics are the same as feature importance.

Scale predictors and inspect convergence before interpreting coefficients. Spark
ML standardizes internally by default, while liblinear does not; reported
coefficients still depend on input units. L1 is useful for compact linear models,
but it may arbitrarily retain one of several correlated predictors and can miss
nonlinear interactions. Strong regularization can validly select no features.

### Permutation Importance

**Supervised evaluation; both backends.** For an already fitted model, permute one
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
variation, not a confidence interval over new datasets. Spark implements distributed
permutations; see the [Spark contract](feature_selection/spark.md#permutation_importance_selection).

## Wrapper methods

### RFE

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

### RFECV

**Supervised; both backends (see their method pages for differences).** RFECV means RFE plus cross-validation used to choose
the feature count. It is a separate sklearn-based selector, not a change to
BrainModelKit's OOT RFE or its persistence contract.

```python
from brainmodelkit.feature_selection.pandas import rfecv

result = rfecv(
    df,
    target_col="target",
    feature_cols=features,
    scoring="roc_auc",
    cv=5,
    step=1,
    min_features_to_select=2,
)
print(result.optimal_feature_count, result.ranking)
print(result.cv_results)
```

Pass a supported named `model` or cloneable importance-producing estimator,
`model_params`, and optionally `n_jobs`/`random_state`. `cv` accepts an integer
or sklearn-compatible splitter/explicit splits; design temporal or grouped
splits yourself. `cv_results` is the installed sklearn version's dictionary,
including mean test scores; newer sklearn versions may add keys. Rank 1 marks
selected features. `model` is the estimator refitted on the selected columns.

RFECV is useful when feature count should be chosen by a CV objective. It is
expensive, and ordinary stratified folds are inappropriate for temporal dependence.
CV used for selection is not an independent final performance estimate. Existing
RFE's OOT metric path is neither calculated nor persisted by this method.
See [sklearn RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html).

### Sequential Feature Selection

**Supervised; both backends (see their method pages for differences).** Forward selection adds the candidate with the best
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

## Robustness / all-relevant methods

### Stability Selection

**Supervised robustness diagnostic; both backends.** Pandas uses repeated stratified bootstrap
samples run a base selector. `selection_frequency = selection_count / n_fits`;
selection requires frequency at least `min_frequency` (default 0.80). This utility
does not claim the formal error bounds of specialized stability-selection methods.

```python
from brainmodelkit.feature_selection.pandas import stability_selection

result = stability_selection(
    df,
    target_col="target",
    feature_cols=features,
    method="feature_importance",
    n_iterations=20,
    sample_fraction=0.8,
    min_frequency=0.80,
    random_state=42,
    selector_kwargs={"model": "random_forest", "top_k": 3},
)
```

`method` is `"feature_importance"` or `"l1"`. Each class is sampled with
replacement using `ceil(class_size * sample_fraction)` rows (at least one).
`sample_fraction` is in (0, 1]. With no `selector_kwargs`, importance selects
the top half (at least one) of features. Otherwise give an explicit `top_k` or
`threshold`; selecting all scores in every fit would produce meaningless frequencies.
For L1, pass `C` and tolerance through `selector_kwargs`. Pass the resampling seed
to this function; configure custom estimator seeds explicitly if needed.

Use `group_by="reference_date"` or a list of group names to run `n_iterations`
within each observed group. Every group must contain the required classes. Groups
have equal weight in overall frequencies regardless of row count; inspect `history`
for group-specific selections. This is within-group resampling, not chronological
train/test validation or cross-group holdout. Missing group keys are retained.

Outputs include `selection_count`, `selection_frequency`, `mean_importance`,
`std_importance`, and `selected` per feature. Means/stds use absolute importance
across all fits, including fits that did not select the feature. History records
the group, iteration, and selected features. Use this to diagnose unstable choices;
correlated interchangeable features may each have low frequency despite strong
joint signal. Runtime scales with iteration and group count, and comparing
coefficients across periods requires consistent scaling.

### Boruta

**Supervised all-relevant selection; both backends.** Boruta
compares real feature importances with randomized shadow features over repeated
fits. It aims to identify all relevant variables rather than a minimal subset.
Pandas delegates to BorutaPy; Spark implements native forest/shadow fitting and
corrected hit tests. Read the [detailed Boruta inference guide](feature_selection/boruta.md)
for the hypotheses, multiplicity rules, and backend differences.

```python
from brainmodelkit.feature_selection.pandas import boruta

result = boruta(
    df,
    target_col="target",
    feature_cols=features,
    model="random_forest",
    model_params={"max_depth": 5},
    n_estimators="auto",
    max_iter=100,
    random_state=42,
)
```

Install `BrainModelKit[pandas,boruta]`. The dependency is loaded only when Boruta
is called; otherwise import/use of the namespace is unaffected. The estimator must
meet BorutaPy's tree importance API requirements. `perc=100` controls the shadow
importance percentile and `alpha=0.05` controls the package's statistical tests.
`max_iter` limits iterations, so tentative decisions may remain.

Outputs are `feature`, `ranking`, `status`, and `selected`; status is `confirmed`,
`tentative`, or `rejected`. Only confirmed features are selected. The returned
`model` is the fitted BorutaPy selector, not a final classifier. Use for exploratory
all-relevant discovery with enough compute, not when an exact feature count or
minimal deployment footprint is required. Importance biases, correlated features,
and small samples still affect decisions. See the [BorutaPy project](https://github.com/scikit-learn-contrib/boruta_py).

## Combining methods

Select from training data and preserve the resulting list for later data:

```python
from brainmodelkit.feature_selection.pandas import (
    completeness,
    correlation_filter,
    information_value,
    rfe,
)

features = completeness(train_df, features, min_completeness=0.70).selected_features
# Encode and impute using training-fitted preprocessing before correlation/RFE.
features = correlation_filter(train_df, features, threshold=0.90).selected_features
features = information_value(
    train_df, "target", features, min_iv=0.02
).selected_features
if len(features) < 3:
    raise ValueError("Too few features remain for the requested RFE final count")
result = rfe(
    "screened_rfe",
    "target",
    train_df,
    oot_df,
    feature_cols=features,
    n_feat_final=3,
)
```

An empty selected list is a legitimate result. Inspect diagnostics and choose an
appropriate policy; downstream selectors reject an empty feature list. Filtering
order matters, especially correlation's input-order representative choice. Do not
automatically tune every cutoff against the final OOT set.

## Pandas workflows

Run [pandas_workflow.py](../examples/feature_selection/pandas_workflow.py) for a
self-contained completeness → correlation → IV → RFE example. It uses synthetic
numeric data and a reproducible holdout; replace that split with the appropriate
temporal design. For more expensive selection, use the screened features with
RFECV or stability selection. Keep one final test set untouched.

## PySpark workflows

Run [pyspark_workflow.py](../examples/feature_selection/pyspark_workflow.py) with
Spark/Java configured. It uses only native Spark selectors and transformations.
Use Spark aggregation/encoding/imputation to prepare inputs, never a driver-side
replica. Grouped completeness and IV diagnostics support `.show()` and distributed
filtering/writing. The existing Spark RFE example remains available separately
and needs a working native persistence environment.

## Backend support matrix

| Method | Pandas | PySpark |
| --- | --- | --- |
| `completeness` (including groups) | Yes | Native aggregations |
| `variance_filter` | Yes | Native population variance |
| `cardinality` | Yes | Native exact distinct counts |
| `correlation_filter` | Pearson/Spearman | Native Spark ML, bounded matrix |
| `mutual_information` | sklearn estimation | Distributed binned counts |
| `chi_square` | Counts/indicators | Native categorical test |
| `anova` | sklearn F-test | Distributed class summaries |
| `information_value` | Yes | Native binning/aggregations |
| `feature_importance_selection` | Yes | Native Spark estimators |
| `l1_selection` | sklearn logistic | Spark ML logistic |
| `permutation_importance_selection` | sklearn permutation | Distributed sort/join permutations |
| `rfe` | Existing API | Existing native API |
| `rfecv` | sklearn RFECV | Fold-local native elimination paths |
| `sequential_selection` | Forward/backward | Native CV candidate search |
| `stability_selection` | Stratified bootstrap/groups | Poisson bootstrap/groups |
| `boruta` | Optional BorutaPy | Native forests and shadow hit tests |

## Scalability considerations

Pandas holds data in process memory and uses sklearn for local model work.
Wrapper and bootstrap costs multiply the underlying fit cost; use cheap screening
first and choose `n_jobs` to avoid excessive parallel memory use.

Spark quality filters aggregate columns together. Exact cardinality can shuffle
many distinct values. Correlation is bounded by feature count rather than row
count; its dense O(p²) matrix is the main driver-memory constraint. Chi-Square
also bounds category count. IV performs per-feature aggregations and numeric
quantile estimation; its category/bin tables remain distributed. Very wide datasets
still generate large query plans and require screening or batches. No method
collects the original training data, creates a local replica, calls `toPandas()`,
or uses Python UDFs to replace Spark SQL expressions. Spark permutation/Boruta
use distributed RDD indexing in addition to SQL sorts/joins; they require classic
Spark rather than Spark Connect. See the [Spark execution guide](feature_selection/spark.md).

Callers control caching: persist a frequently reused, preprocessed Spark frame
when appropriate and unpersist it afterward. These functions do not create Spark
sessions, change caller cache ownership, or promise matching floating-point results
across partition layouts/backends. Native estimator randomness and approximate
quantile boundaries may differ even with fixed seeds.
