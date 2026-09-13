"""Native, distributed PySpark cross-validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from brainmodelkit.metrics.pyspark import auc_gini, ks_ntile

from . import CrossValidationResult


def _validate(df, target_col, feature_cols):
    features = list(feature_cols)
    if not features or len(set(features)) != len(features) or target_col in features:
        raise ValueError(
            "feature_cols must be unique, non-empty, and exclude target_col"
        )
    missing = sorted(set([*features, target_col]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    from pyspark.sql import functions as F

    if (
        df.filter(F.col(target_col).isNull() | ~F.col(target_col).isin(0, 1))
        .limit(1)
        .count()
    ):
        raise ValueError(f"`{target_col}` must contain only binary values 0 and 1.")
    if df.select(target_col).distinct().count() != 2:
        raise ValueError("Cross-validation requires both target classes")
    return features


def _pipeline(features, target, model, params, seed):
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import (
        GBTClassifier,
        LogisticRegression,
        RandomForestClassifier,
    )
    from pyspark.ml.feature import VectorAssembler

    choices = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GBTClassifier,
    }
    if isinstance(model, str):
        if model not in choices:
            raise ValueError(f"Unknown model: {model}")
        estimator = choices[model](**(params or {}))
    else:
        estimator = model.copy({})
        estimator.setParams(**(params or {}))
    for name, value in (
        ("featuresCol", "features"),
        ("labelCol", target),
        ("probabilityCol", "probability"),
        ("predictionCol", "prediction"),
        ("rawPredictionCol", "rawPrediction"),
    ):
        if not estimator.hasParam(name):
            raise TypeError(f"model must expose {name}")
        estimator.set(estimator.getParam(name), value)
    if estimator.hasParam("seed"):
        estimator.set(estimator.getParam("seed"), seed)
    return Pipeline(
        stages=[VectorAssembler(inputCols=features, outputCol="features"), estimator]
    )


def _assign_folds(
    df, target, features, cv, strategy, fold_col, group_col, date_col, seed
):
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    if fold_col:
        if fold_col not in df.columns or fold_col in {*features, target}:
            raise ValueError("fold_col must exist and exclude features and target")
        value = F.col(fold_col)
        bad = (
            df.filter(value.isNull() | ~value.cast("long").isin(list(range(cv))))
            .limit(1)
            .count()
        )
        if bad:
            raise ValueError("fold_col must contain integer fold IDs in [0, cv)")
        return df.withColumn("__bmk_fold", value.cast("int"))
    if strategy in {"group"} or group_col:
        if not group_col or group_col not in df.columns:
            raise ValueError("group_col is required and must exist for group strategy")
        value = F.pmod(F.xxhash64(F.col(group_col), F.lit(seed)), F.lit(cv)).cast("int")
        return df.withColumn("__bmk_fold", value)
    if strategy in {"time", "time_series", "ordered"} or date_col:
        if not date_col or date_col not in df.columns:
            raise ValueError("date_col is required and must exist for time strategy")
        w = Window.orderBy(F.col(date_col), F.monotonically_increasing_id())
        return (
            df.withColumn("__bmk_order", F.row_number().over(w) - 1)
            .withColumn(
                "__bmk_fold",
                F.floor(
                    F.col("__bmk_order") * cv / F.count("*").over(Window.partitionBy())
                ).cast("int"),
            )
            .drop("__bmk_order")
        )
    if strategy not in {"stratified", "kfold"}:
        raise ValueError("strategy must be stratified, kfold, group, or time_series")
    if strategy == "stratified":
        w = Window.partitionBy(target).orderBy(F.rand(seed))
        return df.withColumn(
            "__bmk_fold", F.pmod(F.row_number().over(w) - 1, F.lit(cv)).cast("int")
        )
    w = Window.orderBy(F.rand(seed))
    return df.withColumn(
        "__bmk_fold", F.pmod(F.row_number().over(w) - 1, F.lit(cv)).cast("int")
    )


def cross_validate(
    df,
    target_col: str,
    feature_cols: Sequence[str],
    model: Any,
    model_params: dict | None = None,
    cv: int = 5,
    strategy: str = "stratified",
    group_col: str | None = None,
    date_col: str | None = None,
    fold_col: str | None = None,
    metrics: Sequence[str] = ("ks", "auc", "gini"),
    return_train_metrics: bool = False,
    random_state: int = 42,
    n_tiles: int = 10,
    return_models: bool = False,
) -> CrossValidationResult:
    """Fit Spark ML pipelines on distributed folds; source rows never leave Spark."""
    if not isinstance(cv, int) or isinstance(cv, bool) or cv < 2:
        raise ValueError("cv must be an integer >= 2")
    if not isinstance(n_tiles, int) or isinstance(n_tiles, bool) or n_tiles < 2:
        raise ValueError("n_tiles must be an integer >= 2")
    features = _validate(df, target_col, feature_cols)
    requested = tuple(str(m).lower() for m in metrics)
    if not requested or any(m not in {"ks", "auc", "gini"} for m in requested):
        raise ValueError("metrics must contain only ks, auc, and gini")
    assigned = _assign_folds(
        df,
        target_col,
        features,
        cv,
        strategy,
        fold_col,
        group_col,
        date_col,
        random_state,
    )
    records, models = [], [] if return_models else None
    from pyspark.ml.functions import vector_to_array

    for fold in range(cv):
        train = assigned.filter(assigned.__bmk_fold != fold)
        valid = assigned.filter(assigned.__bmk_fold == fold)
        fitted = _pipeline(features, target_col, model, model_params, random_state).fit(
            train
        )

        def evaluate(frame, fitted_model=fitted):
            scored = fitted_model.transform(frame).withColumn(
                "score", vector_to_array("probability")[1]
            )
            values = {}
            if "ks" in requested:
                values["KS"] = float(
                    ks_ntile(
                        dataframe=scored, target=target_col, n_tiles=n_tiles
                    ).first()["KS"]
                )
            if "auc" in requested or "gini" in requested:
                row = auc_gini(
                    df=scored, target_column=target_col, n_tiles=n_tiles
                ).first()
                if "auc" in requested:
                    values["AUC"] = float(row["auc"])
                if "gini" in requested:
                    values["GINI"] = float(row["gini"])
            return values

        values = {"fold": fold + 1, **evaluate(valid)}
        if return_train_metrics:
            values.update({f"train_{k}": v for k, v in evaluate(train).items()})
        records.append(values)
        if return_models:
            models.append(fitted)
    summary = []
    for metric in [m.upper() for m in requested] + (
        [f"train_{m.upper()}" for m in requested] if return_train_metrics else []
    ):
        values = [r[metric] for r in records]
        finite = [v for v in values if v == v]
        mean = sum(finite) / len(finite) if finite else float("nan")
        summary.append(
            {
                "metric": metric,
                "mean": mean,
                "std": (sum((v - mean) ** 2 for v in finite) / len(finite)) ** 0.5
                if finite
                else float("nan"),
                "min": min(finite) if finite else float("nan"),
                "max": max(finite) if finite else float("nan"),
            }
        )
    return CrossValidationResult(records, summary, models)


__all__ = ["cross_validate"]
