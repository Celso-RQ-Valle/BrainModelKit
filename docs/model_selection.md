# Cross-validation

BrainModelKit cross-validation evaluates models using development data only.
It fits a fresh estimator for every fold, scores validation rows, and returns
fold-level and aggregate metrics. It does not persist models, create MLflow
runs, mutate the input frame, or use an OOT frame.

```python
from brainmodelkit.model_selection.pandas import cross_validate

result = cross_validate(
    train_df, "target", features, "logistic_regression", cv=5,
    strategy="stratified", random_state=42,
)
print(result.fold_metrics)
print(result.summary)
```

Use `strategy="kfold"` for ordinary shuffled folds, `strategy="stratified"`
for binary classification, `group_col`/`strategy="group"` when observations
from one entity must stay together, and `date_col`/`strategy="time_series"`
when temporal ordering must be respected. Spark additionally accepts
`fold_col`, containing integer IDs from `0` through `cv - 1`, so externally
created folds can be consumed without regeneration.

CV is a model-development estimate, not a final fit and not external testing.
Keep `oot_df` separate: select features and parameters using `train_df`, fit
the final model on all of `train_df`, and evaluate that model once on `oot_df`.
Preprocessing that learns state (imputation, encoding, scaling, or selection)
must be inside a fold-fitted pipeline; preprocessing the complete frame first
can leak validation information.

`rfecv` and `sequential_selection` reuse the same CV infrastructure when
evaluating candidate subsets. Their search remains responsible for elimination
or greedy decisions. The result can also be passed directly to future
hyperparameter optimization loops, one parameter candidate at a time.
