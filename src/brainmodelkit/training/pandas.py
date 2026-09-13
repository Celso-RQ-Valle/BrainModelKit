"""Train probability-producing scikit-learn compatible binary classifiers."""

from typing import Literal

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from brainmodelkit.metrics.pandas import _calculate_auc_gini, calculate_ks

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
    save_path: str | None = None,
    save_format: str | None = None,
    save_metadata: bool = True,
):
    """Fit a fresh estimator and return OOT KS/AUC/Gini and class-1 scores.

    ``model`` is logistic_regression, random_forest, gradient_boosting,
    lightgbm, or a cloneable estimator exposing predict_proba. Labels must be
    0/1, with both classes in training. Inputs are not modified. Preprocess
    features before calling, or pass a scikit-learn Pipeline. Signature applies
    to MLflow's native predict output (class labels), not the added score column.
    save_model_to selects folder (default), mlflow, or none. Folder runs save
    the model and reports under output_dir. model_format independently selects
    folder serialization (pickle for Pandas, spark for PySpark by default).
    run_as and runs_as are deprecated aliases for save_model_to.
    signature requires MLflow. Legacy save_path, save_format and mlflow_logging
    are deprecated. save_metadata controls only legacy external model sidecars.
    """
    save_model_to, model_format = resolve_execution(
        "sklearn",
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
    for frame in (train_df, oot_df):
        if not frame[target_col].isin([0, 1]).all():
            raise ValueError("Targets must be non-null binary 0/1 values")
    if train_df[target_col].nunique() != 2:
        raise ValueError("Training requires both target classes")
    if isinstance(model, str):
        choices = {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
        }
        if model == "lightgbm":
            from lightgbm import LGBMClassifier

            choices[model] = LGBMClassifier
        if model not in choices:
            raise ValueError(f"Unknown model: {model}")
        estimator = choices[model](**(model_params or {}))
    else:
        estimator = clone(model).set_params(**(model_params or {}))
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("model must implement predict_proba")
    fitted = estimator.fit(train_df[features], train_df[target_col])
    positive = list(fitted.classes_).index(1)

    def score(frame):
        if frame is None:
            return None
        result = frame.copy()
        result["score"] = (
            fitted.predict_proba(frame[features])[:, positive]
            if len(frame)
            else np.array([], dtype=float)
        )
        return result

    oot = score(oot_df)
    auc, gini = _calculate_auc_gini(oot[target_col], oot["score"])
    metrics = {
        "oot_ks": calculate_ks(oot[target_col], oot["score"]),
        "oot_auc": auc,
        "oot_gini": gini,
    }
    values = getattr(fitted, "feature_importances_", None)
    if values is None and hasattr(fitted, "coef_"):
        values = fitted.coef_[0]
    result = TrainingResult(
        fitted,
        oot,
        score(df_scoring),
        metrics,
        importance_records(features, values),
        None,
    )
    return finalize_run(
        result,
        run_name,
        "sklearn",
        features,
        target_col,
        fitted.get_params(deep=False),
        output_dir,
        save_model_to,
        model_format,
        signature,
        train_df,
        save_path=save_path,
        save_metadata=save_metadata,
    )
