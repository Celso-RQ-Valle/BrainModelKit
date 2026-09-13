# Boruta: shadows, hypotheses, and decisions

[Method index](../feature_selection.md#api-quick-reference) · [Spark execution](spark.md#boruta)

Boruta asks which features consistently beat randomized reference features under
a chosen predictive model. It targets all relevant variables, not a prescribed
number of columns. Relevance is conditional on the data, model, and importance
definition; confirmation does not establish causation. The original method is
described by [Kursa and Rudnicki (2010)](https://www.jstatsoft.org/article/view/v036i11).

## What a shadow feature is

If a real feature is `[10, 20, 20, 40]`, a possible shadow is `[20, 40, 10, 20]`.
Shuffling preserves its marginal distribution but disrupts its row alignment
with the target. Independently permute each active feature each iteration.
Fit a forest on real and shadow columns together so both compete under the same
importance scale. A shadow can have nonzero importance by chance.

Spark keeps confirmed and tentative real features in the forest and removes
rejected features from subsequent fits. It makes at least five independently
shuffled shadows, cycling through active features when fewer than five remain.
With `perc=100`, the reference threshold is the maximum shadow importance.
Other perc values use the linearly interpolated percentile of the shadow vector.
A hit requires real importance **strictly greater** than this threshold; a tie
does not count. Spark uses native random-forest impurity importance.

## The hypothesis test

After t iterations, a feature has h hits. BrainModelKit's Spark implementation
uses the conventional Boruta hit-test reference `H ~ Binomial(t, 0.5)`:

```text
p_accept = P(H >= h) = binomial_survival(h - 1, t, 0.5)
p_reject = P(H <= h) = binomial_cdf(h, t, 0.5)
```

The acceptance alternative is a hit probability above 0.5; the rejection
alternative is below 0.5. These are separate one-sided tests. A feature with
20/20 hits has an uncorrected acceptance p-value of 2^-20 (about 0.000000954).
With 10/20 hits neither tail provides strong evidence. Failing to reject a null
is not validating the null: a feature may remain unresolved because the data or
iteration budget is insufficient.

The p=0.5 reference is an algorithmic convention, not a proof that an irrelevant
real feature beats the maximum of many shadows half the time. Repeated fits share
training rows, and importances are dependent. Thus corrected decisions should
not be presented as guaranteed scientific discovery error control under arbitrary
data. Correlation, class imbalance, forest depth, category cardinality, and
non-exchangeable time/group observations can affect calibration.

## Multiple testing and repeated looks

For Spark `two_step=True` (default), each iteration applies Benjamini–Hochberg
separately to acceptance and rejection p-values across active features. Sort m
p-values, find the largest k satisfying `p_(k) <= alpha*k/m`, and pass the first
k. Each passed feature must additionally satisfy `p <= alpha/t` to account for
repeated looks. Only tentative features change decision; confirmed features
remain confirmed. `two_step=False` instead uses `p <= alpha/p_initial` where
p_initial is the original candidate count. `alpha=0.05` is the default.

Spark exposes both raw tails in `feature_table`; these are not adjusted p-values.
The decision combines raw tails with the correction rules above. The test helper
has analytical tests for all-hit, no-hit, and balanced-hit cases.

The Pandas backend delegates to BorutaPy's default two-step procedure. Its source
also includes postprocessing of unresolved variables and importance-based ranking;
see the [BorutaPy implementation](https://github.com/scikit-learn-contrib/boruta_py/blob/master/boruta/boruta_py.py).

## Stopping and output interpretation

Spark runs up to `max_iter` fits, stopping early when no tentative feature remains.
Only `status='confirmed'` has `selected=True`. `tentative` means unresolved, while
`rejected` means the lower-tail rule fired. Spark does not apply a post-hoc rough
fix to tentative features. It returns status ranks 1/2/3, not a total ordering
within rejected features.

| Field | Meaning |
| --- | --- |
| `feature`, `selected`, `status`, `ranking` | Input name and final decision |
| `hits`, `n_trials` (Spark) | Success count and trials while active |
| `p_accept`, `p_reject` (Spark) | Last raw upper/lower binomial tails |
| `history` (Spark) | Iteration, shadow threshold, real importances, cumulative hits |
| `metadata` (Spark) | Iterations, alpha, perc, corrections, seed, ranking convention |
| `model` | Pandas: BorutaPy selector; Spark: None |

Spark's internal forest contains temporary shadow columns, so it is not returned
as a deployment classifier. Train a new model on `result.selected_features`.
Increasing max_iter can resolve some tentative cases but cannot repair leakage,
unsuitable importance measures, or insufficient independent observations.

## API signatures

### pandas

```python
def boruta(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    oot_df=None,
    df_oot=None,
    df_scoring=None,
    run_name="feature_selection",
    model="random_forest",
    model_params=None,
    n_estimators="auto",
    max_iter=100,
    perc=100,
    alpha=0.05,
    random_state=42,
): ...
```

### pyspark

```python
def boruta(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    model="random_forest",
    model_params=None,
    n_estimators=100,
    max_iter=100,
    perc=100,
    alpha=0.05,
    two_step=True,
    random_state=42,
): ...
```

Pandas additionally accepts optional `oot_df` (alias `df_oot`), `df_scoring`,
and `run_name`. They control final selected-feature evaluation, not shadow tests.
See [evaluation outputs](../feature_selection.md#evaluation-outputs). Spark accepts
the selection arguments shown above and returns diagnostics without a final fit.

## Complete Pandas example

Install `pip install "BrainModelKit[pandas,boruta]"`.

```python
import numpy as np
import pandas as pd
from brainmodelkit.feature_selection.pandas import boruta

rng = np.random.default_rng(42)
x = rng.normal(size=500)
train = pd.DataFrame(
    {
        "signal": x,
        "noise": rng.normal(size=500),
        "target": (x + rng.normal(scale=0.4, size=500) > 0).astype(int),
    }
)
result = boruta(
    train,
    "target",
    ["signal", "noise"],
    n_estimators=100,
    model_params={"max_depth": 5},
    max_iter=30,
    random_state=42,
)
print(pd.DataFrame(result.feature_table))
print(result.selected_features)
```

## Spark example and parameters

Use the [Spark setup](spark.md#example-data), then:

```python
result = fs.boruta(
    train,
    "target",
    features,
    model="random_forest",
    model_params={"maxDepth": 4},
    n_estimators=30,
    max_iter=30,
    perc=100,
    alpha=0.05,
    two_step=True,
    random_state=42,
)
print(result.feature_table)
print(result.history[-1]["shadow_threshold"])
```

`train_df` (alias `df`) supplies nonempty training rows; `target_col` must contain
both 0 and 1. `feature_cols` must be an explicit nonempty list of unique, finite
numeric columns excluding the target. `model='random_forest'` is Spark's only
supported Boruta model. `model_params` uses Spark names; set tree count with the
positive integer `n_estimators=100`, not `numTrees`. `max_iter=100` must be positive;
`0 < perc <= 100`; `0 < alpha < 1`; `random_state=42` seeds forests and permutations.
`two_step` is Boolean. `__bmk_` feature/target names are reserved internally.

Pandas also accepts Boruta-compatible custom tree estimators and
`n_estimators='auto'`; Spark intentionally requires a fixed tree budget. Pandas
does not expose a two_step argument and uses BorutaPy's default. Exact rankings,
unresolved-feature handling, fit-count conventions, random forests, and shadows
differ across backends. Do not expect matching selected lists just from matching
seeds. Spark uses distributed sorts/joins and Python RDD indexing, without
converting data to Pandas, and needs classic Spark workers and executor storage.

## Validating a selection in practice

Develop the selector on training data only. Check feature availability, target
leakage, and preprocessing before interpreting any hits. Inspect sensitivity to
seeds, forest depth, and correlated inputs. Fit a final classifier on the selected
list and evaluate on independent later/group-separated data. A useful validation
question is whether predictions and retained features remain stable in that new
sample, not whether the same training p-value can be made smaller. Retain
tentative diagnostics in the report even when only confirmed features are deployed.
