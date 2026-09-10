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

Each call creates `training_runs/<unique-id>/model_info.json` and
`feature_importance.csv`. Change the parent with `output_dir`. The CSV contains
native tree importance or signed linear coefficients; unsupported custom models
produce a header-only file. These values are not comparable across algorithms.
The folder contains reports, not a serialized model or copies of input data.
The fitted model is returned for prediction or native persistence.

For model persistence and experiment tracking, install `.[mlflow]`, configure
your MLflow tracking URI/experiment, and pass `mlflow_logging=True`. Logging saves
the model, parameters, finite metrics and reports. Existing runs receive a nested
run. `signature=True` infers the native model signature: raw feature inputs and
predicted labels, not the added probability score. Spark logs the full assembler
pipeline so raw numeric features can be supplied again. No remote registration
or tracking server configuration is performed by these functions.
