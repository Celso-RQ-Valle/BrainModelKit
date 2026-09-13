"""Distributed binary classifier training using Spark ML pipelines."""

from typing import Literal

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from brainmodelkit.metrics.pyspark import auc_gini, ks_ntile

from ._common import (
    DEFAULT_DESTINATION,
    TrainingResult,
    finalize_run,
    importance_records,
    resolve_execution,
    validate_columns,
)


def train_model(
    run_name,
    target_col,
    feature_cols,
    train_df,
    oot_df,
    model="logistic_regression",
    *,
    model_params=None,
    df_scoring=None,
    output_dir="training_runs",
    save_model_to: Literal["folder", "mlflow", "none"] = DEFAULT_DESTINATION,
    run_as: str | None = None,
    runs_as: str | None = None,
    model_format: str | None = None,
    mlflow_logging=None,
    signature=False,
    n_tiles=10,
    save_path: str | None = None,
    save_format: str | None = None,
    save_metadata: bool = True,
    overwrite: bool = False,
):
    """Fit assembler + classifier; return distributed scores and tiled OOT metrics.

    Model names match the Pandas backend; lightgbm requires configured SynapseML.
    Custom estimators must expose standard Spark probabilistic classifier columns.
    Features must be numeric and non-null. The returned PipelineModel accepts raw
    feature columns. Only aggregate metrics and feature importances reach the driver.
    MLflow signatures describe native prediction labels. No Spark session is created.
    save_model_to selects folder (default), mlflow, or none. Folder runs save
    the model and reports under output_dir. model_format independently selects
    folder serialization (pickle for Pandas, spark for PySpark by default).
    run_as and runs_as are deprecated aliases for save_model_to.
    signature requires MLflow. Legacy save_path, save_format and mlflow_logging
    are deprecated. save_metadata controls only legacy external model sidecars.
    """
    save_model_to, model_format = resolve_execution(
        "spark",
        save_model_to,
        model_format,
        mlflow_logging,
        save_path,
        save_format,
        signature,
        run_as=run_as,
        runs_as=runs_as,
    )
    features = validate_columns(
        feature_cols,
        target_col,
        [(train_df, True), (oot_df, True), (df_scoring, False)],
    )
    if not isinstance(n_tiles, int) or isinstance(n_tiles, bool) or n_tiles < 2:
        raise ValueError("n_tiles must be an integer >= 2")
    for frame in (train_df, oot_df, df_scoring):
        if frame is not None and set(frame.columns) & {
            "features",
            "prediction",
            "probability",
            "rawPrediction",
        }:
            raise ValueError("Input contains reserved Spark model output columns")
    for frame in (train_df, oot_df):
        label = F.col(target_col)
        if frame.filter(label.isNull() | ~label.isin(0, 1)).limit(1).count():
            raise ValueError("Targets must be non-null binary 0/1 values")
    if train_df.select(target_col).distinct().count() != 2:
        raise ValueError("Training requires both target classes")
    if isinstance(model, str):
        choices = {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GBTClassifier,
        }
        if model == "lightgbm":
            from synapse.ml.lightgbm import LightGBMClassifier

            choices[model] = LightGBMClassifier
        if model not in choices:
            raise ValueError(f"Unknown model: {model}")
        estimator = choices[model](**(model_params or {}))
    else:
        estimator = model.copy({})
        estimator.setParams(**(model_params or {}))
    for name, value in {
        "featuresCol": "features",
        "labelCol": target_col,
        "probabilityCol": "probability",
        "predictionCol": "prediction",
        "rawPredictionCol": "rawPrediction",
    }.items():
        if not estimator.hasParam(name):
            raise TypeError(f"model must expose {name}")
        estimator.set(estimator.getParam(name), value)
    fitted = Pipeline(
        stages=[VectorAssembler(inputCols=features, outputCol="features"), estimator]
    ).fit(train_df)

    def score(frame):
        if frame is None:
            return None
        return fitted.transform(frame).withColumn(
            "score", vector_to_array("probability")[1]
        )

    oot = score(oot_df)
    auc = auc_gini(df=oot, target_column=target_col, n_tiles=n_tiles).first()
    ks = ks_ntile(dataframe=oot, target=target_col, n_tiles=n_tiles).first()["KS"]
    metrics = {
        "oot_ks": float(ks),
        "oot_auc": float(auc["auc"]),
        "oot_gini": float(auc["gini"]),
    }
    classifier = fitted.stages[-1]
    values = getattr(classifier, "featureImportances", None)
    if values is None:
        values = getattr(classifier, "coefficients", None)
    if values is None and hasattr(classifier, "getFeatureImportances"):
        values = classifier.getFeatureImportances()
    result = TrainingResult(
        fitted,
        oot,
        score(df_scoring),
        metrics,
        importance_records(features, values),
        None,
    )
    params = {p.name: v for p, v in estimator.extractParamMap().items()}
    return finalize_run(
        result,
        run_name,
        "spark",
        features,
        target_col,
        params,
        output_dir,
        save_model_to,
        model_format,
        signature,
        train_df,
        save_path=save_path,
        save_metadata=save_metadata,
        overwrite=overwrite,
    )
