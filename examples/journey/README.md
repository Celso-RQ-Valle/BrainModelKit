# Connected modeling walkthroughs

[Documentation home](../../docs/README.md) ? [Spark journey](../../README.md#spark-guide) ? [pandas journey](../../README.md#pandas-guide)

Run from a repository checkout with the appropriate backend extra installed:

```bash
python examples/journey/pandas_workflow.py
python examples/journey/spark_workflow.py
```

Both use [sample_data.py](sample_data.py): 120 synthetic accounts, one reference
row each, and up to three earlier observations per account. The class-conditional
numeric signal is deliberately synthetic, not a performance benchmark. April,
May, and June cohorts have 40 distinct accounts each. The helper is not a public
library API. Replace its output with real histories and a labeled spine while
preserving the documented schemas and point-in-time availability rules.

The scripts build features and latency, retain only generated numeric inputs,
fill missing values with a fixed demonstration value of zero, and split by
reference date. They fit variance selection only on April. They compare two
logistic-regression configurations on May, freeze the subset and configuration,
refit on April + May, and evaluate June OOT. They then report metrics, risk tiles,
and segment cubes, and score rows with the same raw-feature schema.

The training API calls May validation `oot_df` and reports `oot_*` metric keys;
this naming does not make May a final evaluation set. June is never passed into
selection or tuning. Fixed zero filling learns no state; choose a meaningful
production missing-value policy and fit any learned preprocessing within each
training split. The selected feature order must accompany the deployed model.

Expected structural results: 40 predictions for June, finite class-1 probabilities
between 0 and 1, scalar metric dictionaries, four overall risk tiles, and cube
rows for `large`, `small`, and overall `Geral`. Numeric scores are environment-
and backend-dependent; Spark tiling and its estimator differ from pandas.

## Persist and reload

These standalone variants save on the final fit, load the resulting model, and
assert that rescoring produces valid probabilities:

```bash
python examples/journey/pandas_workflow.py --save-dir training_runs/pandas_journey
python examples/journey/spark_workflow.py --save-dir training_runs/spark_journey
```

The pandas path uses pickle and the Spark path uses a PipelineModel loader.
Only load trusted artifacts. The Spark script pins local Python workers to its
interpreter and uses two local threads for reproducibility of the demo setup;
production sessions remain caller-managed. Spark folder persistence requires a
working Hadoop filesystem, including the [Windows prerequisites](../../docs/setup.md#spark-on-windows).
Default runs use no persistence, so missing filesystem helpers do not block
training/evaluation. MLflow and alternative formats are documented in the
[training reference](../../docs/training.md).

## CV and Optuna alternatives

The main scripts intentionally preserve chronological development validation.
Use the [separate CV/optimization walkthrough](../../docs/journey-validation.md)
for the verified public APIs and the distinction between fixed-feature CV and
fold-fitted preprocessing/selection. They do not automatically provide a
leakage-safe end-to-end pipeline for previously selected features.
