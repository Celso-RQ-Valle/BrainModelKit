"""Native Spark ML importance and L1 selection."""

import math
from collections.abc import Sequence
from typing import Any

from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.sql import DataFrame

from brainmodelkit.training._common import importance_records
from brainmodelkit.training.pyspark import train_model

from ._evaluation import evaluation_frame, finalize_selection
from ._selection import SelectionResult, number, ranked, resolve_frame
from ._spark_evaluation import validate_evaluation
from ._spark_statistics import supervised, vector_frame


def feature_importance_selection(
    train_df: DataFrame | None = None,
    target_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: DataFrame | None = None,
    oot_df: DataFrame | None = None,
    df_oot: DataFrame | None = None,
    df_scoring: DataFrame | None = None,
    run_name: str = "feature_selection",
    n_tiles: int = 10,
    model: Any = "random_forest",
    model_params: dict | None = None,
    threshold: float | None = None,
    top_k: int | None = None,
    random_state: int = 42,
) -> SelectionResult:
    """Fit a Spark classifier and rank its native importance vector.

    See [Feature Importance](../../docs/feature_selection.md#feature-importance)
    for supported models and examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    oot_df = evaluation_frame(oot_df, df_oot, df_scoring)
    validate_evaluation(train_df, oot_df, df_scoring, target_col, feature_cols)
    features = supervised(train_df, target_col, feature_cols)
    choices = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GBTClassifier,
    }
    if isinstance(model, str):
        if model == "lightgbm":
            try:
                from synapse.ml.lightgbm import LightGBMClassifier
            except ImportError as exc:
                raise ImportError(
                    "Spark LightGBM requires configured SynapseML and JVM packages"
                ) from exc
            choices[model] = LightGBMClassifier
        if model not in choices:
            raise ValueError(f"Unknown model: {model}")
        fitted_estimator = choices[model]()
    else:
        fitted_estimator = model.copy({})
    if fitted_estimator.hasParam("seed"):
        fitted_estimator.set(fitted_estimator.getParam("seed"), random_state)
    fitted_estimator.setParams(**(model_params or {}))
    fitted_estimator.set(fitted_estimator.getParam("featuresCol"), "features")
    fitted_estimator.set(fitted_estimator.getParam("labelCol"), "label")
    fitted = fitted_estimator.fit(vector_frame(train_df, features, target_col))
    values = getattr(fitted, "featureImportances", None)
    coefficients = getattr(fitted, "coefficients", None)
    if values is None:
        values = coefficients
    if values is None and hasattr(fitted, "getFeatureImportances"):
        values = fitted.getFeatureImportances()
    if (
        values is None
        or len(values) != len(features)
        or any(not math.isfinite(v) for v in values)
    ):
        raise ValueError("model must expose one finite native importance per feature")
    records = {r["feature"]: r for r in importance_records(features, values)}
    rows = [records[f] for f in features]
    if coefficients is not None:
        for i, row in enumerate(rows):
            row["coefficients"] = [float(coefficients[i])]
    result = ranked(
        rows, "importance", "feature_importance", threshold, top_k, absolute=True
    )
    result.model = fitted
    return finalize_selection(
        result,
        train_model,
        train_df=train_df,
        oot_df=oot_df,
        target_col=target_col,
        feature_cols=features,
        model=fitted_estimator,
        df_scoring=df_scoring,
        run_name=run_name,
        n_tiles=n_tiles,
    )


def l1_selection(
    train_df: DataFrame | None = None,
    target_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: DataFrame | None = None,
    oot_df: DataFrame | None = None,
    df_oot: DataFrame | None = None,
    df_scoring: DataFrame | None = None,
    run_name: str = "feature_selection",
    n_tiles: int = 10,
    reg_param: float = 0.1,
    coefficient_tolerance: float = 1e-8,
    max_iter: int = 100,
) -> SelectionResult:
    """Select non-zero coefficients from native Spark L1 logistic regression.

    See [L1 Selection](../../docs/feature_selection.md#l1-selection) for Spark
    regularization semantics and examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    oot_df = evaluation_frame(oot_df, df_oot, df_scoring)
    validate_evaluation(train_df, oot_df, df_scoring, target_col, feature_cols)
    number(reg_param, "reg_param")
    number(coefficient_tolerance, "coefficient_tolerance")
    result = feature_importance_selection(
        train_df,
        target_col,
        feature_cols,
        model="logistic_regression",
        model_params={
            "elasticNetParam": 1.0,
            "regParam": reg_param,
            "maxIter": max_iter,
        },
    )
    result.method = "l1"
    result.metadata.update(
        reg_param=reg_param, coefficient_tolerance=coefficient_tolerance
    )
    for row in result.feature_table:
        row["selected"] = abs(row["importance"]) > coefficient_tolerance
    return finalize_selection(
        result,
        train_model,
        train_df=train_df,
        oot_df=oot_df,
        target_col=target_col,
        feature_cols=feature_cols,
        model=LogisticRegression(
            elasticNetParam=1.0, regParam=reg_param, maxIter=max_iter
        ),
        df_scoring=df_scoring,
        run_name=run_name,
        n_tiles=n_tiles,
    )
