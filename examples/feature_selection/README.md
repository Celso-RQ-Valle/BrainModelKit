# Recursive feature elimination

Run from the repository root. Edit `model`, `model_params`, `step`, and
`n_feat_final` in the scripts to configure selection.

## Spark

With Java/Spark configured:

```bash
python -m pip install -e ".[pyspark]"
python examples/feature_selection/spark_example.py
```

## Pandas

```bash
python -m pip install -e ".[pandas]"
python examples/feature_selection/pandas_example.py
```

Both examples write iteration reports and selected features under
`feature_selection_runs/`. They use random holdouts for demonstration; use a
later-period OOT dataset for your own evaluation. See the main README for model
options, importance conventions and optional MLflow persistence.
