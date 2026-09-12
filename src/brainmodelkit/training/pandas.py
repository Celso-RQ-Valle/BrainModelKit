"""Train probability-producing scikit-learn compatible binary classifiers."""

from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from brainmodelkit.metrics.pandas import _calculate_auc_gini, calculate_ks
from brainmodelkit.persistence._common import PYTHON_FORMATS, validate_format
from brainmodelkit.persistence.python import _save_python_model

from ._common import TrainingResult, finish, importance_records, validate_columns


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
    mlflow_logging=False,
    signature=False,
    save_path: str | None = None,
    save_format: str = "pickle",
    save_metadata: bool = True,
):
    """Fit a fresh estimator and return OOT KS/AUC/Gini and class-1 scores.

    ``model`` is logistic_regression, random_forest, gradient_boosting,
    lightgbm, or a cloneable estimator exposing predict_proba. Labels must be
    0/1, with both classes in training. Inputs are not modified. Preprocess
    features before calling, or pass a scikit-learn Pipeline. Signature applies
    to MLflow's native predict output (class labels), not the added score column.
    Local reports are stored under a unique child of output_dir.
    If save_path is provided, persist the fitted model separately using save_format
    (pickle by default). Optional backends require their corresponding extras.
    Metadata is written to <save_path>.metadata.json when save_metadata is true.
    """
    validate_format(save_format, PYTHON_FORMATS)
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
    if signature and not mlflow_logging:
        raise ValueError("signature requires mlflow_logging=True")
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
        Path(output_dir),
    )
    if save_path is not None:
        _save_python_model(
            fitted,
            save_path,
            save_format,
            save_metadata=save_metadata,
            target_col=target_col,
            feature_cols=features,
            parameters=fitted.get_params(deep=False),
            input_example=train_df[features].head(1) if save_format == "onnx" else None,
        )
    return finish(
        result,
        run_name,
        "sklearn",
        features,
        target_col,
        fitted.get_params(deep=False),
        output_dir,
        mlflow_logging,
        signature,
        train_df,
    )
