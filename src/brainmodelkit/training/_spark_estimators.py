"""One estimator/pipeline factory for training, CV, and optimization."""


def validate_backend(lgbm_backend, params=None):
    if lgbm_backend not in {"default", "synapseml"}:
        raise ValueError("lgbm_backend must be default or synapseml")
    if "lgbm_backend" in (params or {}):
        raise ValueError("Pass lgbm_backend as a backend selector, not in model_params")


def is_default_lightgbm(model, lgbm_backend):
    return (
        isinstance(model, str)
        and model in {"lightgbm", "lgbm"}
        and lgbm_backend == "default"
    )


def make_estimator(model, params, target, seed=42, lgbm_backend="default"):
    validate_backend(lgbm_backend, params)
    params = dict(params or {})
    if is_default_lightgbm(model, lgbm_backend):
        from ._spark_lightgbm import SparkLightGBMEstimator, require_sparklightgbm

        require_sparklightgbm()
        # These names belong to SynapseML, not the native LightGBM API. Native
        # LightGBM otherwise silently ignores unknown parameters.
        if any(any(c.isupper() for c in name) for name in params):
            raise ValueError(
                "SparkLightGBM uses snake_case/native LightGBM parameters; use "
                "num_iterations, learning_rate, num_leaves, etc."
            )
        params.setdefault("seed", seed)
        params.update(
            features_col="features",
            label_col=target,
            probability_col="probability",
            prediction_col="prediction",
            raw_prediction_col="rawPrediction",
            leaf_prediction_col=None,
        )
        return SparkLightGBMEstimator(params)

    if isinstance(model, str):
        from pyspark.ml.classification import (
            DecisionTreeClassifier,
            GBTClassifier,
            LogisticRegression,
            RandomForestClassifier,
        )

        choices = {
            "logistic_regression": LogisticRegression,
            "decision_tree": DecisionTreeClassifier,
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GBTClassifier,
        }
        if model in {"lightgbm", "lgbm"}:
            try:
                from synapse.ml.lightgbm import LightGBMClassifier
            except ImportError as exc:
                raise ImportError(
                    "lgbm_backend='synapseml' requires "
                    "'brainmodelkit[synapseml]' and the matching SynapseML JVM "
                    "package attached before Spark starts"
                ) from exc
            choices[model] = LightGBMClassifier
        if model not in choices:
            raise ValueError(f"Unknown model: {model}")
        try:
            estimator = choices[model](**params)
        except TypeError as exc:
            if model not in {"lightgbm", "lgbm"} or "JavaPackage" not in str(exc):
                raise
            raise RuntimeError(
                "Load the matching SynapseML Maven package for your Spark/Scala "
                "runtime before starting Spark, then restart the session. The "
                "Python synapseml wrapper alone is insufficient."
            ) from exc
    else:
        estimator = model.copy({})
        estimator.setParams(**params)
    for name, value in {
        "featuresCol": "features",
        "labelCol": target,
        "probabilityCol": "probability",
        "predictionCol": "prediction",
        "rawPredictionCol": "rawPrediction",
    }.items():
        if not estimator.hasParam(name):
            raise TypeError(f"model must expose {name}")
        estimator.set(estimator.getParam(name), value)
    if (
        estimator.hasParam("seed")
        and "seed" not in params
        and (isinstance(model, str) or not estimator.isSet(estimator.getParam("seed")))
    ):
        estimator.set(estimator.getParam("seed"), seed)
    return estimator


def make_pipeline(features, target, model, params, seed=42, lgbm_backend="default"):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler

    return Pipeline(
        stages=[
            VectorAssembler(inputCols=features, outputCol="features"),
            make_estimator(model, params, target, seed, lgbm_backend),
        ]
    )
