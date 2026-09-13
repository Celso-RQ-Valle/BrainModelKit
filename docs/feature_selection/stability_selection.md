# Stability Selection: stability_selection

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pandas as fs` for Pandas,
or `from brainmodelkit.feature_selection import pyspark as fs` for Spark.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

**Spark implementation:** read the [Spark algorithm and execution contract](spark.md#stability_selection).
The explanations below describe shared concepts and the Pandas behavior. Consult
the signatures for backend-specific arguments; they are not interchangeable.

## Signatures

### Pandas

```python
def stability_selection(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    oot_df=None,
    df_oot=None,
    df_scoring=None,
    run_name="feature_selection",
    method="feature_importance",
    n_iterations=20,
    sample_fraction=0.8,
    min_frequency=0.8,
    random_state=42,
    group_by=None,
    selector_kwargs=None,
): ...
```

### PySpark

```python
def stability_selection(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    method="feature_importance",
    n_iterations=20,
    sample_fraction=0.8,
    min_frequency=0.8,
    random_state=42,
    group_by=None,
    selector_kwargs=None,
    max_groups=100,
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
| `method` | Algorithm variant: correlation accepts pearson/spearman; stability accepts feature_importance/l1. |
| `n_iterations` | Positive number of stability resamples/fits per observed group. |
| `sample_fraction` | Sampling fraction in (0,1]. Pandas uses class-stratified fixed-size bootstrap; Spark uses Poisson multiplicities with this mean. |
| `min_frequency` | Inclusive stability selection frequency threshold in [0,1]. |
| `random_state` | Seed for supported random estimators/resampling. Spark partition/order changes may still change results. |
| `group_by` | Optional grouping column name or list, including observed null groups. Controls grouped diagnostics or separate stability fits. |
| `selector_kwargs` | Base stability selector configuration; specify top_k or threshold for importance. Outer resampling controls must not be placed here. |
| `max_groups` | Positive Spark limit on collected distinct group keys before stability fitting. |

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
`python examples/feature_selection/method_example.py --method stability_selection`
from the repository root for a self-contained Pandas example. Add `--backend spark`
for the Spark implementation. See the method-specific parameter tables above.

## How it works

**Supervised robustness diagnostic; Pandas behavior.** Repeated stratified bootstrap
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

## Spark usage

See [the complete Spark setup and stability_selection example](spark.md#stability_selection) for the
distributed algorithm, parameters, result fields, limits, and interpretation.
