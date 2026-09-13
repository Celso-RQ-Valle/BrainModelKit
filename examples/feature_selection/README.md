# Feature selection examples

For method explanations, assumptions, and the backend matrix, see the
[Feature Selection Guide](../../docs/feature_selection.md).

- `pandas_workflow.py`: completeness → correlation → IV → existing RFE.
- `pyspark_workflow.py`: native completeness → variance → correlation → IV → importance.
- `pandas_example.py` and `spark_example.py`: original RFE examples, retained unchanged.

Run workflows with `python examples/feature_selection/pandas_workflow.py` or
`python examples/feature_selection/pyspark_workflow.py`. The Spark workflow keeps
all dataset rows distributed and does not require model persistence.

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
later-period OOT dataset for your own evaluation. See the guide for model
options, importance conventions and optional MLflow persistence.
