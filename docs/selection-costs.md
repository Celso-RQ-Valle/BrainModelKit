# Choosing selectors by work and purpose

[Documentation home](README.md) ? [Selection contracts](feature_selection.md)

Both namespaces export all sixteen selectors below. Tiers describe expected work
from the current implementations, not measured timings. Within a tier there is
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
| `cardinality` | Distinct values | Low?moderate | Exact distinct counts and shuffles | Before encoding | [Method](feature_selection/cardinality.md) |
| `anova` | Class-mean signal | Low?moderate | Two class summaries per feature | Numeric screening | [Method](feature_selection/anova.md) |
| `chi_square` | Count/category association | Low?moderate | Categorical contingency tables | Counts (pandas); category codes (Spark) | [Method](feature_selection/chi_square.md) |
| `information_value` | Binary distribution separation | Moderate | Quantiles and grouped bin counts | Inspect WoE bins | [Method](feature_selection/information_value.md) |
| `mutual_information` | Nonlinear univariate signal | Moderate | Quantile binning and joint-count shuffles | Nonlinear screening | [Method](feature_selection/mutual_information.md) |
| `correlation_filter` | Reduce redundancy | Moderate?high | Distributed matrix/ranks; p? driver memory | After quality checks | [Method](feature_selection/correlation_filter.md) |
| `l1_selection` | Sparse linear subset | Moderate?high | Distributed logistic iterations | Compact linear models | [Method](feature_selection/l1_selection.md) |
| `feature_importance_selection` | Model-based ranking | Moderate?high | One native model fit | Capture model-specific signal | [Method](feature_selection/feature_importance_selection.md) |
| `permutation_importance_selection` | Held-out reliance | High | Repeats ? sorting, joins, predictions | Validate model reliance | [Method](feature_selection/permutation_importance_selection.md) |
| `rfe` | Recursive elimination | High | Distributed fits per step; reports | Conditional importance | [Method](feature_selection/rfe.md) |
| `stability_selection` | Selection repeatability | High?very high | Poisson resample fits ? groups | Check robustness | [Method](feature_selection/stability_selection.md) |
| `rfecv` | Choose subset size | Very high | Folds ? elimination fits plus refit | Budgeted subset search | [Method](feature_selection/rfecv.md) |
| `sequential_selection` | Greedy subset search | Very high | Candidate subsets ? distributed fits | Small candidate pools | [Method](feature_selection/sequential_selection.md) |
| `boruta` | All-relevant selection | Very high | Shadow sorts/joins + repeated forests | All-relevant hypothesis testing | [Method](feature_selection/boruta.md) |

## pandas comparison

| Method | Purpose | Typical relative cost | Main cost driver | When to use | Guide |
| --- | --- | --- | --- | --- | --- |
| `completeness` | Availability | Low | Non-null scans; optional grouped aggregates | Missing-input screening | [Method](feature_selection/completeness.md) |
| `variance_filter` | Remove constants | Low | Column moments | Before fitting | [Method](feature_selection/variance_filter.md) |
| `cardinality` | Distinct values | Low?moderate | Exact per-column distinct sets | Before encoding | [Method](feature_selection/cardinality.md) |
| `anova` | Class-mean signal | Low?moderate | Class-wise sums and variances | Numeric screening | [Method](feature_selection/anova.md) |
| `chi_square` | Count/category association | Low?moderate | Count-by-class summaries | Counts (pandas); category codes (Spark) | [Method](feature_selection/chi_square.md) |
| `information_value` | Binary distribution separation | Moderate | Binning and class counts | Inspect WoE bins | [Method](feature_selection/information_value.md) |
| `mutual_information` | Nonlinear univariate signal | Moderate | Nearest-neighbor estimation for continuous data | Nonlinear screening | [Method](feature_selection/mutual_information.md) |
| `correlation_filter` | Reduce redundancy | Moderate?high | Feature-pair matrix; Spearman ranks | After quality checks | [Method](feature_selection/correlation_filter.md) |
| `l1_selection` | Sparse linear subset | Moderate?high | Iterative penalized logistic fit | Compact linear models | [Method](feature_selection/l1_selection.md) |
| `feature_importance_selection` | Model-based ranking | Moderate?high | One estimator fit | Capture model-specific signal | [Method](feature_selection/feature_importance_selection.md) |
| `permutation_importance_selection` | Held-out reliance | High | Features ? repeats ? predictions | Validate model reliance | [Method](feature_selection/permutation_importance_selection.md) |
| `rfe` | Recursive elimination | High | Fits per elimination step; reports | Conditional importance | [Method](feature_selection/rfe.md) |
| `stability_selection` | Selection repeatability | High?very high | Bootstrap fits ? groups | Check robustness | [Method](feature_selection/stability_selection.md) |
| `rfecv` | Choose subset size | Very high | Folds ? elimination fits | Budgeted subset search | [Method](feature_selection/rfecv.md) |
| `sequential_selection` | Greedy subset search | Very high | Candidate subsets ? folds | Small candidate pools | [Method](feature_selection/sequential_selection.md) |
| `boruta` | All-relevant selection | Very high | Forests on real + shadow columns | All-relevant hypothesis testing | [Method](feature_selection/boruta.md) |

## Evidence and execution limits

The inventory is the `__all__` lists in the two public feature-selection modules.
Cost drivers were checked against `_pandas_quality`, `_pandas_statistics`,
`_pandas_model`, `_pandas_wrappers`, and the corresponding `_spark_*` modules.
No timings or hardware-independent rankings are claimed.

Spark does not convert selector source tables to pandas. It does collect scalar
or feature-sized summaries: correlation uses an O(p?) driver matrix, ANOVA
collects class summaries, MI collects bounded levels/counts and bin edges, and
stability collects bounded group keys. Quality aggregates also return summaries
to Python. Permutation and Boruta need classic Spark Python workers and RDD
indexing, so they do not support Spark Connect. Neither this inventory nor a
successful local run certifies Spark Connect support for every other selector.
See [Spark algorithm contracts](feature_selection/spark.md).

Pandas chi-square uses nonnegative integer counts/indicators; Spark uses integer
category codes. Pandas MI uses sklearn estimation; Spark MI uses binned counts.
Pandas L1 uses `C`; Spark uses `reg_param`. Search/refit costs, scorer definitions,
bootstrap semantics, and estimator parameter spelling also differ. Preserve the
method-specific feature order and inspect diagnostics, including empty selections.
