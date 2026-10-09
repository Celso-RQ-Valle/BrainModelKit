# Hyperparameter optimization

Install `brainmodelkit[pandas,optimization]` or
`brainmodelkit[pyspark,optimization]`. Both optimization modules expose the same
`objective_metric`, `direction`, `target_metric`, `target_limit`, and
`target_operator` arguments. The three target arguments must be supplied together;
omit all three for ordinary unconstrained optimization (CV KS maximization by
default). `metrics` selects the base metrics to calculate, defaulting to all three.

```python
from brainmodelkit.optimization.pandas import optimize
# For Spark: from brainmodelkit.optimization.pyspark import optimize

space = {"num_leaves": {"type": "int", "low": 8, "high": 64}}
result = optimize(
    train_df,
    oot_df,
    "target",
    features,
    "lightgbm",
    space,
    objective_metric="oot_ks",
    direction="maximize",
    target_metric="train_oot_ks_abs_shift",
    target_limit=0.05,
    target_operator="<=",
    n_trials=50,
)

# Alternatively, minimize instability while requiring OOT KS of at least 40%:
stable = optimize(
    train_df,
    oot_df,
    "target",
    features,
    "lightgbm",
    space,
    objective_metric="train_oot_ks_abs_shift",
    direction="minimize",
    target_metric="oot_ks",
    target_limit=0.40,
    target_operator=">=",
    n_trials=50,
)
print(result.constraint_satisfied, result.selected_trial_number)
print(result.best_params, result.best_value)
print(result.trials)
```

For each `m` in `auc`, `gini`, and `ks`, objectives and constraints can use:

| Name | Meaning |
| --- | --- |
| `train_m` | In-sample performance of a model fitted on all training rows |
| `cv_m` | Mean performance on held-out CV folds, not training-fold performance |
| `oot_m` | Full-training model evaluated on OOT |
| `m` | Backward-compatible alias for `cv_m` |
| `m_shift`, `cv_oot_m_shift` | OOT minus CV (signed) |
| `train_oot_m_shift` | OOT minus in-sample training (signed) |
| `cv_oot_m_abs_shift` | `abs(cv_m - oot_m)` |
| `train_oot_m_abs_shift` | `abs(train_m - oot_m)` |

Substitute the metric name, for example `cv_auc`, `oot_gini`, or
`train_oot_ks_abs_shift`. Values use decimals: KS 40% is `0.40`; a gap of five
percentage points is `0.05`. Absolute shifts are neither relative changes nor
PSI. Spark uses the existing tiled metric utilities (`n_tiles` controls their
approximation); only aggregate metrics reach the driver, with no conversion of
the dataframes to pandas.

Constrained TPE receives a signed constraint residual on every completed trial,
so after its startup exploration it learns to favor feasibility throughout the
search. Feasible trials always beat infeasible trials. Among feasible trials,
selection optimizes the objective in `direction`; among infeasible trials it
minimizes the violation, using the objective only to break ties. There is no
weighted combination of objective and constraint.

If nothing is feasible, optimization emits a warning, sets
`constraint_satisfied=False`, and refits the least-violating trial. This does
**not** mean the target was achieved. `selected_trial_number`, `best_params`,
`best_value`, `best_model`, and the saved model refer to that selected trial.
Use these result fields rather than assuming `study.best_trial` provides the
least-violating fallback across Optuna versions. `best_value` is the original
objective metric, not a penalty. Trial records include metrics, residual,
nonnegative violation, feasibility, completion status, and actual estimator
parameters (including fixed values returned by callable search specifications).
Invalid/missing/non-finite metrics raise an error rather than silently satisfying
a constraint.

**OOT used as an objective or constraint becomes a tuning validation set.** Keep
a separate untouched temporal holdout for final evaluation. Without an OOT
objective or constraint, OOT remains diagnostic and the default selection uses
only CV. In-sample training metrics are not estimates of generalization.

## Spark LightGBM

`train_model`, `cross_validate`, and `optimize` accept
`lgbm_backend="default"` (SparkLightGBM) or `"synapseml"`. It is an execution
choice, never a model hyperparameter: do not put it in `model_params` or
`search_space`. `model="lightgbm"` and Spark's alias `model="lgbm"` both work.
The same backend and parameters are used for CV, full-training metric
evaluation, and final refitting. Pandas continues using native `LGBMClassifier`;
custom Spark estimators continue to be copied and configured normally.

For an existing managed Spark installation:

```bash
pip install "brainmodelkit[sparklightgbm,optimization]"
# Or explicitly choose SynapseML:
pip install "brainmodelkit[synapseml,optimization]"
```

SparkLightGBM 0.9.2 uses `LightGBMClassifier(features_col=..., label_col=...,
**native_lightgbm_parameters)`, including `num_iterations`, `learning_rate`,
`num_leaves`, and `num_workers`. SynapseML uses its own Spark parameter names,
such as `numIterations`, `learningRate`, and `numLeaves`, and requires matching
JVM artifacts configured before Spark starts. Missing dependencies never cause
an automatic backend switch. See the verified
[SparkLightGBM public API](https://pypi.org/project/sparklightgbm/0.9.2/).

SparkLightGBM requires classic Spark with RDD/barrier access and the Python
package, NumPy, and compatible native LightGBM installed on **every executor**.
Spark Connect is unsupported. Its automatic `num_workers=None` uses one worker
on local Spark, collecting training data into driver NumPy arrays. Set
`num_workers>1` for distributed native training; each worker shard must fit in
executor memory and barrier tasks need concurrent slots and reachable ports.
Match `num_threads` to executor CPU allocation. BrainModelKit does not collect
Spark data into pandas.

BrainModelKit adapts the bridge to a Python Spark ML stage and converts its
probability/raw-score arrays to Spark vectors. Class-1 probability is `score`.
Returned pipelines accept raw feature columns. Folder persistence uses the
normal Spark `PipelineModel` writer/reader, embedding native LightGBM text and
bridge metadata; BrainModelKit and SparkLightGBM must be importable when loading.
The upstream native model alone is not a Spark pipeline. The adapter implements
Spark ML readable/writable interfaces, so the existing folder and MLflow Spark
persistence paths can save the Python pipeline. Both require a working Hadoop
filesystem (including MLflow's DFS staging). For loading in another environment,
install the BrainModelKit version containing this adapter plus SparkLightGBM
and compatible LightGBM on the driver and executors; MLflow dependency inference
alone cannot guarantee executor dependencies.
