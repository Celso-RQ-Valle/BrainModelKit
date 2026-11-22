# Hyperparameter optimization

[Documentation home](README.md) ? [Backend journeys](../README.md)

Install `brainmodelkit[pandas,optimization]` or
`brainmodelkit[pyspark,optimization]`. Both optimization modules expose the same
`objective_metric`, `direction`, `target_metric`, `target_limit`, and
`target_operator` arguments. The three target arguments must be supplied together;
omit all three for ordinary unconstrained optimization (CV KS maximization by
default). `metrics` selects the base metrics to calculate, defaulting to all three.

```python
from brainmodelkit.optimization.pandas import optimize
# For Spark: from brainmodelkit.optimization.pyspark import optimize

space = {"num_leaves": [8, 64]}
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
    save_trials_path="reports/optimization_trials.csv",
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

## Search spaces

See [models and hyperparameters](models/README.md) for each supported model,
including pandas, SparkLightGBM, and SynapseML parameter names.

Use `[minimum, maximum]` intervals in `search_space`:

```python
search_space = {"min_data_in_leaf": [200, 1000], "num_leaves": [15, 63]}
```

Two integer bounds sample integers throughout the inclusive interval, so
`[200, 1000]` can produce 537. If either bound is a float, the optimizer samples
a float, for example `"learning_rate": [0.01, 0.1]`. Bounds must be finite,
numeric, and ordered. Use `[100]` to fix a parameter. Single-value lists also
accept strings, booleans, or `None`, for example `["gini"]`, `[True]`, or
`[None]`. Empty lists, lists with more than two elements, non-finite numbers,
and non-numeric interval bounds are rejected. Equal numeric bounds remain
accepted for compatibility; prefer `[value]` for fixed settings. This replaces the earlier candidate-list interpretation.

Sampling is linear by default. Existing dictionary ranges and callables remain
supported: use `{"type": "float", "low": 0.001, "high": 10.0, "log": True}`
for logarithmic sampling. For discrete choices, use an explicit dictionary such
as `{"type": "categorical", "choices": ["gini", "entropy"]}`.
Optuna samples within the trial budget; it does not guarantee every value or
combination is tested. `best_params` contains chosen scalar values for refitting.

## Save and compare trials

Both backends accept `save_trials_path="reports/optimization_trials.csv"`.
The default `None` writes no trial file. This is independent of `save_as` and
`save_path`, which control model persistence. The CSV is written on the local
driver filesystem, creates missing parent directories, and overwrites an existing
file at that path. It is saved after trial selection and before the final refit.
`result.trials_path` returns the saved path.

Each row contains the trial number, status, objective `value`, one `param_<name>`
column per hyperparameter, train/CV/validation metrics and their shifts,
constraint residual/violation, `feasible`, and `selected`. The `params` column
stores the complete parameter dictionary as JSON, preserving types for refitting.
The `selected` flag identifies the library's choice; it does not prevent you from
choosing another completed configuration. Only requested base metrics and their
derived shifts are included (KS, AUC, and Gini by default).

```python
# Add this argument to the optimize call shown above:
# save_trials_path="reports/optimization_trials.csv"
import json
import pandas as pd

trials_df = pd.read_csv(result.trials_path)
# Here validation AUC is the comparison metric; avoid selecting on final OOT.
comparison = trials_df.loc[
    trials_df["status"].eq("COMPLETE") & trials_df["feasible"]
].sort_values("oot_auc", ascending=False)
if comparison.empty:
    raise ValueError("No feasible configuration; inspect the trial metrics and limits.")
chosen_params = json.loads(comparison.iloc[0]["params"])
# Use chosen_params as train_model(..., model_params=chosen_params),
# keeping the same estimator/pipeline, backend, and feature order.
```

The in-memory `result.trials` remains a pandas DataFrame in the pandas backend
and a list of records in Spark; the CSV loads as a DataFrame for either backend.
The export contains trial summaries only, never training or validation rows.

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
