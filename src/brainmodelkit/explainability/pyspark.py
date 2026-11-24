"""Worker-local SHAP for exportable Spark models; explicit bounded local mode."""

from copy import deepcopy

from brainmodelkit._probability import CapabilityError, choice, outputs
from brainmodelkit._probability_spark import column
from brainmodelkit.training import TrainingResult


def _export(model, features):
    """Export only known objects; never send a JVM model to Python workers."""
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import LogisticRegressionModel
    from pyspark.ml.feature import VectorAssembler

    from brainmodelkit.training._spark_lightgbm import SparkLightGBMModel

    model = model.model if isinstance(model, TrainingResult) else model
    if isinstance(model, PipelineModel):
        if len(model.stages) != 2 or not isinstance(model.stages[0], VectorAssembler):
            raise CapabilityError(
                "Spark explanation supports VectorAssembler + classifier only; "
                "other preprocessing stages need a validated export adapter"
            )
        assembler, estimator = model.stages
        if assembler.getInputCols() != features:
            raise ValueError("Feature order differs from fitted VectorAssembler")
        model = estimator
    if isinstance(model, LogisticRegressionModel):
        if model.numClasses != 2 or model.numFeatures != len(features):
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
    raise CapabilityError(
        "Unsupported Spark model object. Supported: Spark LogisticRegressionModel "
        "and BrainModelKit SparkLightGBMModel, optionally after VectorAssembler. "
        "SynapseML is not implemented; JVM models cannot run inside Python workers."
    )


def _restore(spec, features):
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


class PredictionPipeline:
    """Compose existing Spark model predictions with SQL calibration and scoring."""

    def __init__(
        self,
        model,
        feature_cols: list[str],
        *,
        event_label=1,
        calibrator=None,
        score_aligner=None,
    ):
        self.model = model.model if isinstance(model, TrainingResult) else model
        self.feature_cols = list(feature_cols)
        self.event_label = event_label
        if event_label not in (0, 1):
            raise ValueError("Supported Spark classifiers have class labels 0 and 1")
        _export(self.model, self.feature_cols)
        self.calibrator, self.score_aligner = (
            deepcopy(calibrator),
            deepcopy(score_aligner),
        )
        if calibrator is not None and calibrator.event_label != event_label:
            raise ValueError("Calibrator event_label differs from model event")

    def transform(self, df, *, overwrite: bool = False):
        from uuid import uuid4

        from pyspark.ml.functions import vector_to_array

        from brainmodelkit.calibration.pyspark import LogisticCalibrator
        from brainmodelkit.scoring.pyspark import ScoreAligner

        names = ["event_probability"]
        if self.calibrator is not None:
            names += [
                "calibrated_event_probability",
                "calibrated_non_event_probability",
            ]
        if self.score_aligner is not None:
            names += ["business_score", "business_score_unrounded"]
        outputs(df.columns, names, overwrite)
        original = [name for name in df.columns if name not in names]
        predicted = self.model.transform(df)
        predicted = predicted.withColumn(
            "event_probability",
            vector_to_array(column("probability"))[int(self.event_label)],
        )
        probability = "event_probability"
        if self.calibrator is not None:
            cal = LogisticCalibrator.from_dict(self.calibrator.to_dict())
            cal.output_probability = "event"
            temp = "__bmk_" + uuid4().hex
            if cal.input_space == "raw":
                value = vector_to_array(column("rawPrediction"))[int(self.event_label)]
            else:
                value = (
                    1 - column(probability)
                    if cal.input_probability == "non_event"
                    else column(probability)
                )
            predicted = cal.transform(
                predicted.withColumn(temp, value),
                temp,
                output_col="calibrated_event_probability",
                non_event_col="calibrated_non_event_probability",
                overwrite=overwrite,
            ).drop(temp)
            probability = "calibrated_event_probability"
        if self.score_aligner is not None:
            align = ScoreAligner.from_dict(self.score_aligner.to_dict())
            align.input_probability = "event"
            predicted = align.transform(predicted, probability, overwrite=overwrite)
            if align.decimals is None:
                predicted = predicted.withColumn(
                    "business_score_unrounded", column("business_score")
                )
        return predicted.select(*[column(name) for name in original + names])


class ExplanationResult:
    """Distributed wide explanation frame with all original-feature contributions."""

    def __init__(self, frame, metadata: dict, features: list[str], keys: list[str]):
        self.frame, self.metadata, self.features, self.keys = (
            frame,
            metadata,
            features,
            keys,
        )

    def to_frame(self):
        return self.frame

    def global_importance(self):
        from pyspark.sql import functions as F

        summary = self.frame.agg(
            *[
                F.avg(F.abs(column("contribution::" + name))).alias(name)
                for name in self.features
            ]
        )
        return (
            summary.select(
                F.explode(
                    F.array(
                        *[
                            F.struct(
                                F.lit(name).alias("feature"),
                                column(name).alias("mean_absolute_contribution"),
                            )
                            for name in self.features
                        ]
                    )
                ).alias("item")
            )
            .select("item.*")
            .orderBy(F.desc("mean_absolute_contribution"))
        )

    def aggregate_groups(self, groups: dict[str, str]):
        from pyspark.sql import functions as F

        if set(groups) - set(self.features):
            raise ValueError("Unknown grouped features")
        grouped = {}
        for feature in self.features:
            group = groups.get(feature, feature)
            grouped[group] = grouped.get(group, F.lit(0.0)) + column(
                "contribution::" + feature
            )
        return self.frame.select(
            *[column(name) for name in self.keys],
            *[value.alias("group::" + name) for name, value in grouped.items()],
        )

    def top_contributors(self, top_k: int = 5):
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be positive")
        rows = self.frame.select(
            *[column(name) for name in self.keys],
            F.explode(
                F.array(
                    *[
                        F.struct(
                            F.lit(name).alias("feature"),
                            column("contribution::" + name).alias("contribution"),
                            column("value::" + name).alias("value"),
                        )
                        for name in self.features
                    ]
                )
            ).alias("item"),
        ).select(*[column(name) for name in self.keys], "item.*")
        rows = rows.where("contribution != 0").withColumn(
            "direction",
            F.when(F.col("contribution") > 0, "positive").otherwise("negative"),
        )
        window = Window.partitionBy(
            *[column(name) for name in self.keys], "direction"
        ).orderBy(F.desc(F.abs(F.col("contribution"))), "feature")
        return rows.withColumn("rank", F.row_number().over(window)).where(
            F.col("rank") <= top_k
        )

    def local(self, max_rows: int = 1):
        """Explicit bounded collection for waterfall/summary; filter frame first."""
        import numpy as np
        import pandas as pd

        from ._result import ExplanationResult as LocalResult

        if type(max_rows) is not int or max_rows < 1:
            raise ValueError("max_rows must be positive")
        frame = self.frame.limit(max_rows + 1).toPandas()
        if len(frame) > max_rows:
            raise ValueError(
                "Result exceeds max_rows; filter by observation keys first"
            )
        values = pd.DataFrame({name: frame["value::" + name] for name in self.features})
        contributions = pd.DataFrame(
            {name: frame["contribution::" + name] for name in self.features}
        )
        return LocalResult(
            frame.prediction.to_numpy(),
            frame.base_value.to_numpy(),
            contributions,
            values,
            frame[self.keys],
            self.metadata,
            frame.residual.to_numpy(),
            frame.tolerance.to_numpy(),
            np.asarray(frame.rounded_prediction)
            if "rounded_prediction" in frame
            else None,
        )


class ModelExplainer:
    """Explicit distributed mapInPandas or bounded local execution.

    Spark models are exported on the driver, then reconstructed once per partition.
    No nested Spark calls occur in workers. Numeric scalar features only.
    """

    def __init__(
        self,
        model,
        background,
        feature_cols: list[str],
        *,
        key_cols: list[str],
        execution: str = "distributed",
        max_background: int = 100,
        max_rows: int = 1000,
        seed: int = 42,
        **options,
    ):
        from pyspark.sql import functions as F

        from brainmodelkit._probability_spark import numeric
        from brainmodelkit.calibration.pandas import LogisticCalibrator
        from brainmodelkit.scoring.pandas import ScoreAligner

        from .pandas import ModelExplainer as LocalExplainer

        choice(execution, ("distributed", "local"), "execution")
        if any(
            type(value) is not int or value < 1 for value in (max_background, max_rows)
        ):
            raise ValueError("max_background and max_rows must be positive")
        if not key_cols or len(set(key_cols)) != len(key_cols):
            raise ValueError(
                "Spark explanations require unique key_cols (composite keys supported)"
            )
        self.features, self.keys = list(feature_cols), list(key_cols)
        if list(background.columns) != self.features:
            raise ValueError("Background must contain exactly ordered features")
        self.schema = [
            (name, background.schema[name].dataType) for name in self.features
        ]
        for name in self.features:
            numeric(background, name)
        self.spec = _export(model, self.features)
        # Content-hash sample, bounded before collection; keys are not features.
        selected = background.orderBy(
            F.xxhash64(F.lit(seed), *[column(name) for name in self.features]),
            *[column(name) for name in self.features],
        ).limit(max_background)
        self.background = selected.select(
            *[column(name).cast("double").alias(name) for name in self.features]
        ).toPandas()
        for name, cls in (
            ("calibrator", LogisticCalibrator),
            ("score_aligner", ScoreAligner),
        ):
            if options.get(name) is not None:
                options[name] = cls.from_dict(options[name].to_dict())
        self.options = {
            **options,
            "key_cols": self.keys,
            "max_background": max_background,
            "max_rows": max_rows,
            "seed": seed,
            "class_labels": [0, 1],
        }
        self.execution, self.max_rows = execution, max_rows
        self.prototype = LocalExplainer(
            _restore(self.spec, self.features),
            self.background,
            self.features,
            **self.options,
        )
        self.prototype._initialize()

    def explain(self, df):
        from pyspark.sql import functions as F
        from pyspark.sql.types import BooleanType, DoubleType, StructField, StructType

        from brainmodelkit._probability_spark import numeric

        from .pandas import ModelExplainer as LocalExplainer

        if (
            len(df.columns) != len(set(df.columns))
            or set(df.columns) != set(self.features + self.keys)
            or [name for name in df.columns if name not in self.keys] != self.features
        ):
            raise ValueError(
                "Frame must contain exactly ordered features and configured keys"
            )
        for name, dtype in self.schema:
            if df.schema[name].dataType != dtype:
                raise TypeError("Feature dtypes differ from background")
            numeric(df, name)
        if (
            df.where(
                F.greatest(
                    *[column(name).isNull().cast("int") for name in self.keys], F.lit(0)
                )
                > 0
            )
            .limit(1)
            .count()
        ):
            raise ValueError("Observation keys must not be null")
        if (
            df.groupBy(*[column(name) for name in self.keys])
            .count()
            .where("count > 1")
            .limit(1)
            .count()
        ):
            raise ValueError("Composite observation keys must be unique")
        selected = df.select(
            *[column(name) for name in self.keys],
            *[column(name).cast("double").alias(name) for name in self.features],
        )
        if self.execution == "local":
            local = selected.limit(self.max_rows + 1).toPandas()
            return self.prototype.explain(local)
        fields = [df.schema[name] for name in self.keys]
        for name in ("prediction", "base_value", "residual", "tolerance"):
            fields.append(StructField(name, DoubleType(), False))
        fields.append(StructField("reconstruction_ok", BooleanType(), False))
        if (
            self.prototype.target == "business_score"
            and self.prototype.pipeline.score_aligner.decimals is not None
        ):
            fields += [
                StructField(name, DoubleType(), False)
                for name in ("rounded_prediction", "rounding_difference")
            ]
        for name in self.features:
            fields += [
                StructField("contribution::" + name, DoubleType(), False),
                StructField("value::" + name, DoubleType(), False),
            ]
        spec, features, background, options = (
            self.spec,
            self.features,
            self.background,
            self.options,
        )

        # Capture portable state only, never self/prototype/JVM model/SparkSession.
        def batches(iterator):
            explainer = LocalExplainer(
                _restore(spec, features), background, features, **options
            )
            for batch in iterator:
                for start in range(0, len(batch), options["max_rows"]):
                    yield explainer.explain(
                        batch.iloc[start : start + options["max_rows"]]
                    ).to_frame()

        result = selected.mapInPandas(batches, StructType(fields))
        metadata = {
            **self.prototype.pipeline.metadata(),
            "target": self.prototype.target,
            "units": "score points"
            if self.prototype.target == "business_score"
            else "raw margin (log-odds)"
            if self.prototype.target == "raw"
            else "probability (0 to 1)",
            "reference": self.prototype.reference,
            "method": self.prototype.method,
            "max_evals": self.prototype.max_evals,
            "seed": self.prototype.seed,
            "feature_dictionary": self.prototype.feature_dictionary,
            "execution": "distributed mapInPandas",
            "adapter": self.spec["kind"],
            "atol": self.prototype.atol,
            "rtol": self.prototype.rtol,
        }
        return ExplanationResult(result, metadata, self.features, self.keys)
