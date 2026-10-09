"""Distributed binary classifier training using Spark ML pipelines."""

from typing import Literal

from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from brainmodelkit.cube_analysis.pyspark import calculate_metrics as calculate_cube
from brainmodelkit.metrics.pyspark import auc_gini, ks_ntile
from brainmodelkit.persistence.spark import validate_spark_persistence

from ._common import (
    DEFAULT_DESTINATION,
    TrainingResult,
    finalize_run,
    importance_records,
    resolve_execution,
    validate_columns,
)
from ._spark_estimators import is_default_lightgbm, make_pipeline, validate_backend


def train_model(
    run_name,
    target_col,
    feature_cols,
    train_df,
    oot_df,
    model="logistic_regression",
    *,
    model_params=None,
    lgbm_backend="default",
    random_state=42,
    df_scoring=None,
    output_dir="training_runs",
    save_model_to: Literal["folder", "mlflow", "none"] = DEFAULT_DESTINATION,
    run_as: str | None = None,
    runs_as: str | None = None,
    model_format: str | None = None,
    mlflow_logging=None,
    signature=False,
    mlflow_dfs_tmp: str | None = None,
    n_tiles=10,
    save_path: str | None = None,
    save_format: str | None = None,
    save_metadata: bool = True,
    overwrite: bool = False,
    analysis_cube=None,
):
    """Fit assembler + classifier; return distributed scores and tiled OOT metrics.

    LightGBM defaults to SparkLightGBM; lgbm_backend="synapseml" selects SynapseML.
    Custom estimators must expose standard Spark probabilistic classifier columns.
    Features must be numeric and non-null. The returned PipelineModel accepts raw
    feature columns. Metrics use distributed Spark utilities. SparkLightGBM's
    num_workers=1 (including automatic local mode) materializes training rows in
    driver NumPy memory; use num_workers>1 for distributed native training.
    MLflow signatures describe native prediction labels. No Spark session is created.
    save_model_to selects folder (default), mlflow, or none. Folder runs save
    the model and reports under output_dir. model_format independently selects
    folder serialization (pickle for Pandas, spark for PySpark by default).
    run_as and runs_as are deprecated aliases for save_model_to.
    signature is used only with MLflow; folder options are ignored elsewhere.
    mlflow_dfs_tmp optionally supplies MLflow Spark's dfs_tmpdir. None leaves
    MLflow defaults/environment configuration unchanged; ignored outside MLflow.
    Legacy save_path, save_format and mlflow_logging are deprecated.
    save_metadata controls only legacy external model sidecars.
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
    validate_backend(lgbm_backend, model_params)
    features = validate_columns(
        feature_cols,
        target_col,
        [(train_df, True), (oot_df, True), (df_scoring, False)],
    )
    validate_spark_persistence(
        train_df.sparkSession,
        save_model_to,
        path=save_path if save_path is not None else output_dir,
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
    pipeline = make_pipeline(
        features, target_col, model, model_params, random_state, lgbm_backend
    )
    estimator = pipeline.getStages()[-1]
    fitted = pipeline.fit(train_df)

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
    cube = (
        calculate_cube(
            df=oot,
            group_columns=analysis_cube,
            target_column=target_col,
            n_tiles=n_tiles,
        )
        if analysis_cube is not None
        else None
    )
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
        analysis_cube=cube,
    )
    params = {p.name: v for p, v in estimator.extractParamMap().items()}
    if is_default_lightgbm(model, lgbm_backend):
        params = dict(params["options"])
    if isinstance(model, str) and model in {"lightgbm", "lgbm"}:
        params["lgbm_backend"] = lgbm_backend
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
        mlflow_dfs_tmp=mlflow_dfs_tmp,
    )
