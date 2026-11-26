"""Spark classifier output contracts and portable worker representations."""

from brainmodelkit._probability import CapabilityError


class SparkAdapter:
    """Keep the complete fitted pipeline on the driver for native transforms."""

    def __init__(self, model):
        self.estimator = model.stages[-1] if hasattr(model, "stages") else model
        self.options = (
            self.estimator.getOrDefault(self.estimator.options)
            if hasattr(self.estimator, "nativeModel")
            else {}
        )

    @property
    def probability_col(self):
        if hasattr(self.estimator, "getProbabilityCol"):
            return self.estimator.getProbabilityCol()
        return self.options.get("probability_col", "probability")

    @property
    def features_col(self):
        if hasattr(self.estimator, "getFeaturesCol"):
            return self.estimator.getFeaturesCol()
        if hasattr(self.estimator, "nativeModel"):
            return self.options.get("features_col", "features")
        return None

    @property
    def raw_col(self):
        from pyspark.ml.classification import LogisticRegressionModel

        from brainmodelkit.training._spark_lightgbm import SparkLightGBMModel

        if isinstance(self.estimator, SparkLightGBMModel):
            return self.options.get("raw_prediction_col", "rawPrediction")
        if (
            isinstance(self.estimator, LogisticRegressionModel)
            and self.estimator.getFamily() != "multinomial"
        ):
            return self.estimator.getRawPredictionCol()
        raise CapabilityError(
            "Raw calibration requires a verified event-oriented margin"
        )


def export_predictor(model, features):
    """Export only known objects; never send a JVM model to Python workers."""
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import LogisticRegressionModel
    from pyspark.ml.feature import VectorAssembler

    from brainmodelkit.training._spark_lightgbm import SparkLightGBMModel

    if isinstance(model, PipelineModel):
        if len(model.stages) != 2 or not isinstance(model.stages[0], VectorAssembler):
            raise CapabilityError(
                "Spark explanation supports VectorAssembler + classifier only; "
                "other preprocessing stages need a validated export adapter"
            )
        assembler, estimator = model.stages
        if assembler.getInputCols() != features:
            raise ValueError("Feature order differs from fitted VectorAssembler")
        if assembler.getOutputCol() != SparkAdapter(estimator).features_col:
            raise CapabilityError(
                "Export requires the classifier to consume the assembled vector"
            )
        model = estimator
    if isinstance(model, LogisticRegressionModel):
        if (
            model.getFamily() == "multinomial"
            or model.numClasses != 2
            or model.numFeatures != len(features)
        ):
            raise CapabilityError(
                "Only binary Spark logistic regression with matching "
                "scalar numeric inputs is supported"
            )
        return {
            "kind": "logistic",
            "coefficients": model.coefficients.toArray().tolist(),
            "intercept": float(model.intercept),
        }
    if isinstance(model, SparkLightGBMModel):
        return {
            "kind": "lightgbm",
            "native_model": model.getOrDefault(model.nativeModel),
        }
    if type(model).__module__.startswith("synapse.ml.lightgbm."):
        from synapse.ml.lightgbm import LightGBMClassificationModel

        if isinstance(model, LightGBMClassificationModel):
            return {"kind": "synapseml", "native_model": model.getNativeModel()}
    raise CapabilityError(
        "Unsupported Spark model object. Supported: Spark LogisticRegressionModel "
        "and BrainModelKit SparkLightGBMModel, optionally after VectorAssembler. "
        "JVM models cannot run inside Python workers."
    )


def restore_predictor(spec, features):
    import numpy as np

    if spec["kind"] == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        model.coef_ = np.array([spec["coefficients"]])
        model.intercept_ = np.array([spec["intercept"]])
        model.n_features_in_ = len(features)
        model.feature_names_in_ = np.array(features)
        return model
    from lightgbm import Booster

    model = Booster(model_str=spec["native_model"])
    if (
        model.num_feature() != len(features)
        or model.num_model_per_iteration() != 1
        or not str(model.dump_model().get("objective", "")).startswith("binary")
    ):
        raise CapabilityError(
            "SparkLightGBM export requires a binary objective and matching features"
        )
    return model
