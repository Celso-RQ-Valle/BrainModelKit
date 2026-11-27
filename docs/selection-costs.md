# Choosing selectors by work and purpose

[Documentation home](README.md) / [Selection contracts](feature_selection.md)

Both namespaces export all sixteen selectors below.

Cost tiers are qualitative estimates. Runtime depends on row count, feature count,
cardinality, model choice, iterations, repeats, CV folds, and execution resources.
Here p is the number of input features. A full feature-pair matrix uses O(p^2)
memory, which can limit a Spark driver independently of runtime. Within a tier there is
no speed ranking. A small forest can be cheaper than a very wide correlation
matrix; distribution adds scheduling, network, shuffle, and storage costs.

Suggested progression: remove unusable inputs; inspect redundancy and univariate
signal where appropriate; consider model-based selection; spend on iterative
methods only when justified; evaluate a frozen subset on untouched data. This is
guidance, not a requirement to run every method.

## Spark comparison

| Method | Purpose | Typical relative cost | Main cost driver | When to use | Guide |
| --- | --- | --- | --- | --- | --- |
| `completeness` | Availability | Low | Non-null aggregates; grouped shuffles | Missing-input screening | [Method](feature_selection/completeness.md) |
| `variance_filter` | Remove constants | Low | Native var_pop aggregates | Before fitting | [Method](feature_selection/variance_filter.md) |
| `cardinality` | Distinct values | Low to moderate | Exact distinct counts and shuffles | Before encoding | [Method](feature_selection/cardinality.md) |
| `anova` | Class-mean signal | Low to moderate | Two class summaries per feature | Numeric screening | [Method](feature_selection/anova.md) |
| `chi_square` | Count/category association | Low to moderate | Categorical contingency tables | Integer category codes | [Method](feature_selection/chi_square.md) |
| `information_value` | Binary distribution separation | Moderate | Quantiles and grouped bin counts | Inspect WoE bins | [Method](feature_selection/information_value.md) |
| `mutual_information` | Nonlinear univariate signal | Moderate | Quantile binning and joint-count shuffles | Nonlinear screening | [Method](feature_selection/mutual_information.md) |
| `correlation_filter` | Reduce redundancy | Moderate to high | Distributed correlation (ranking for Spearman); O(p^2) driver memory | After quality checks | [Method](feature_selection/correlation_filter.md) |
| `l1_selection` | Sparse linear subset | Moderate to high | Distributed logistic iterations | Compact linear models | [Method](feature_selection/l1_selection.md) |
| `feature_importance_selection` | Model-based ranking | Moderate to high | One native model fit | Capture model-specific signal | [Method](feature_selection/feature_importance_selection.md) |
| `permutation_importance_selection` | Held-out reliance | High | Repeated sorting, joins, and predictions per feature | Validate model reliance | [Method](feature_selection/permutation_importance_selection.md) |
| `rfe` | Recursive elimination | High | Distributed fits per step; reports | Conditional importance | [Method](feature_selection/rfe.md) |
| `stability_selection` | Selection repeatability | High to very high | Model fits on Poisson resamples, per group when grouped | Check robustness | [Method](feature_selection/stability_selection.md) |
| `rfecv` | Choose subset size | Very high | Full-data elimination, CV fits per subset, then elimination again to the chosen size | Budgeted subset search | [Method](feature_selection/rfecv.md) |
| `sequential_selection` | Greedy subset search | Very high | Distributed CV fits for candidate subsets, then a final fit | Small candidate pools | [Method](feature_selection/sequential_selection.md) |
| `boruta` | All-relevant selection | Very high | Shadow sorts/joins + repeated forests | All-relevant hypothesis testing | [Method](feature_selection/boruta.md) |

## pandas comparison

| Method | Purpose | Typical relative cost | Main cost driver | When to use | Guide |
| --- | --- | --- | --- | --- | --- |
| `completeness` | Availability | Low | Non-null scans; optional grouped aggregates | Missing-input screening | [Method](feature_selection/completeness.md) |
| `variance_filter` | Remove constants | Low | Column moments | Before fitting | [Method](feature_selection/variance_filter.md) |
| `cardinality` | Distinct values | Low to moderate | Exact per-column distinct sets | Before encoding | [Method](feature_selection/cardinality.md) |
| `anova` | Class-mean signal | Low to moderate | Class-wise sums and variances | Numeric screening | [Method](feature_selection/anova.md) |
| `chi_square` | Count/category association | Low to moderate | Count-by-class summaries | Nonnegative counts or indicators | [Method](feature_selection/chi_square.md) |
| `information_value` | Binary distribution separation | Moderate | Binning and class counts | Inspect WoE bins | [Method](feature_selection/information_value.md) |
| `mutual_information` | Nonlinear univariate signal | Moderate | Nearest-neighbor estimation for continuous data | Nonlinear screening | [Method](feature_selection/mutual_information.md) |
| `correlation_filter` | Reduce redundancy | Moderate to high | Local O(p^2) correlation matrix; ranking for Spearman | After quality checks | [Method](feature_selection/correlation_filter.md) |
| `l1_selection` | Sparse linear subset | Moderate to high | Iterative penalized logistic fit | Compact linear models | [Method](feature_selection/l1_selection.md) |
| `feature_importance_selection` | Model-based ranking | Moderate to high | One estimator fit | Capture model-specific signal | [Method](feature_selection/feature_importance_selection.md) |
| `permutation_importance_selection` | Held-out reliance | High | Shuffle one local column and rescore per feature and repeat | Validate model reliance | [Method](feature_selection/permutation_importance_selection.md) |
| `rfe` | Recursive elimination | High | Fits per elimination step; reports | Conditional importance | [Method](feature_selection/rfe.md) |
| `stability_selection` | Selection repeatability | High to very high | Model fits on class-stratified bootstrap samples, per group when grouped | Check robustness | [Method](feature_selection/stability_selection.md) |
| `rfecv` | Choose subset size | Very high | Full-data elimination, CV fits per subset, then one refit on the chosen subset | Budgeted subset search | [Method](feature_selection/rfecv.md) |
| `sequential_selection` | Greedy subset search | Very high | Local CV fits for candidate subsets, then a final fit | Small candidate pools | [Method](feature_selection/sequential_selection.md) |
| `boruta` | All-relevant selection | Very high | Forests on real + shadow columns | All-relevant hypothesis testing | [Method](feature_selection/boruta.md) |

## Evidence and execution limits

The inventory is the `__all__` lists in the two public feature-selection modules.
Cost drivers were checked against `_pandas_quality`, `_pandas_statistics`,
`_pandas_model`, `_pandas_wrappers`, and the corresponding `_spark_*` modules.
No timings or hardware-independent rankings are claimed.

Spark does not convert selector source tables to pandas. It does collect summaries of different sizes: correlation collects the full O(p^2) matrix, ANOVA
collects class summaries, MI collects bounded levels/counts and bin edges, and
stability collects bounded group keys. Quality aggregates also return summaries
to Python. Permutation and Boruta need classic Spark Python workers and RDD
indexing, so they do not support Spark Connect. Neither this inventory nor a
successful local run certifies Spark Connect support for every other selector.
See [Spark algorithm contracts](feature_selection/spark.md).

Spark chi-square uses integer category codes and pandas uses nonnegative integer counts or indicators.
Spark MI uses binned counts and pandas uses sklearn estimation.
Spark L1 uses `reg_param` and pandas uses `C`. Search/refit costs, scorer definitions,
bootstrap semantics, and estimator parameter spelling also differ. Preserve the
method-specific feature order and inspect diagnostics, including empty selections.
