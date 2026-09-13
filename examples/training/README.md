# Training examples

From the repository root, install `python -m pip install -e ".[pandas]"`
and execute `python examples/training/pandas_example.py`. For Spark, install
`.[pyspark]`, configure Java/Spark, and run `python examples/training/spark_example.py`.

Edit the arguments in either script to describe your model, features and target.
Both use random holdouts for demonstration; supply an actual later-period dataset
as `oot_df` for out-of-time evaluation. Data preprocessing is the caller's responsibility.

Model names: `logistic_regression` (default), `random_forest`,
`gradient_boosting`, `lightgbm`. Parameters use each backend's native names.
You can also pass an estimator instance through `model`. Pandas clones it; Spark
copies it. Both train fresh models. Pandas LightGBM needs `.[pandas,lightgbm]`;
Spark LightGBM needs SynapseML installed with matching JVM packages in your Spark
session. Installing the Python LightGBM extra does not configure SynapseML.

Results expose `model`, `oot_predictions`, `scoring_predictions`, `metrics`,
`feature_importance`, `output_dir`, and optional MLflow `run_id`. The `score`
column is the probability of target 1. Training requires both 0 and 1 labels;
OOT may be single-class, producing undefined metrics. Spark KS/AUC are tiled
approximations controlled by `n_tiles`; Pandas metrics are exact.

With the default `save_model_to="folder"`, each call creates `training_runs/<run_name>_<run_id>/` containing
the fitted model, `model_info.json`, `metrics.json`, `metadata.json`, and
`feature_importance.csv`. Change the parent with `output_dir`.
Pandas defaults to `model_format="pickle"`; Spark uses `model_format="spark"`.
The result exposes `save_model_to`, `model_format`, `run_id`, `output_dir`, and `model_uri`.
Use `save_model_to="none"` for training without artifacts.

For experiment tracking, install `.[mlflow]`, configure your MLflow tracking
URI/experiment, and pass `save_model_to="mlflow"`. Model, parameters, finite metrics,
metadata and reports go to MLflow; `output_dir` is then `None`.
Existing active runs receive a nested run. `signature=True` optionally infers
raw feature inputs and predicted labels, not the added probability score.
Spark logs the full assembler pipeline. No tracking server configuration is
performed by these functions.

See [destination, format and migration details](../../README.md#model-training).
