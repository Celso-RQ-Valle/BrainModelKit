"""Native Spark prediction and SQL transformations."""

from copy import deepcopy

from brainmodelkit._probability import CapabilityError, outputs
from brainmodelkit._probability_spark import column
from brainmodelkit.persistence import resolve_model

from . import Workflow
from .spark_adapters import SparkAdapter


class PredictionPipeline(Workflow):
    """Compose existing Spark model predictions with SQL calibration and scoring."""

    def __init__(
        self,
        model,
        feature_cols: list[str] | None = None,
        *,
        event_label=None,
        calibrator=None,
        score_aligner=None,
        probability_col=None,
    ):
        if isinstance(model, PredictionPipeline):
            probability_col = probability_col or model.probability_col
            event_label = model.event_label if event_label is None else event_label
            calibrator = model.calibrator if calibrator is None else calibrator
            score_aligner = (
                model.score_aligner if score_aligner is None else score_aligner
            )
        if event_label is None:
            event_label = 1
        resolved = resolve_model(model, feature_cols=feature_cols)
        if resolved.class_labels is not None and resolved.class_labels != [0, 1]:
            raise ValueError("Spark probability vectors require class indices [0, 1]")
        self.probability_col = probability_col
        self.schema = resolved.schema
        self._inherit_metadata(model if isinstance(model, Workflow) else None)
        self.model = resolved.model
        self.adapter = SparkAdapter(self.model)
        self.feature_cols = resolved.feature_cols or []
        if not self.feature_cols or len(set(self.feature_cols)) != len(
            self.feature_cols
        ):
            raise ValueError("Provide unique ordered feature_cols")
        for config in (calibrator, score_aligner):
            if (
                config
                and config.feature_order
                and config.feature_order != self.feature_cols
            ):
                raise ValueError(
                    "Configuration feature_order differs from model inputs"
                )
        self.event_label = event_label
        if event_label not in (0, 1):
            raise ValueError("Supported Spark classifiers have class labels 0 and 1")

        self.calibrator, self.score_aligner = (
            deepcopy(calibrator),
            deepcopy(score_aligner),
        )
        if calibrator is not None:
            calibrator._ready()
            if calibrator.event_label != event_label:
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
        from pyspark.sql import functions as F

        if len(set(df.columns)) != len(df.columns) or set(self.feature_cols) - set(
            df.columns
        ):
            raise ValueError("Missing features or duplicate input columns")
        for name, dtype in (self.schema or {}).items():
            if df.schema[name].dataType.simpleString() != dtype:
                raise TypeError(f"Expected Spark input {name!r} to have type {dtype!r}")
        if getattr(self.adapter.estimator, "numClasses", 2) != 2:
            raise CapabilityError("Only binary Spark classifiers are supported")
        probability_col = self.probability_col or self.adapter.probability_col
        if (
            not hasattr(self.model, "stages")
            and self.adapter.features_col is not None
            and self.adapter.features_col not in self.feature_cols
        ):
            raise ValueError("Standalone classifier requires its fitted vector column")
        source = "__bmk_" + uuid4().hex
        inputs = df.select(
            F.struct(*[column(n) for n in original]).alias(source),
            *[column(n) for n in self.feature_cols],
        )
        predicted = self.model.transform(inputs)
        if probability_col not in predicted.columns:
            raise CapabilityError(
                "Spark classifier must produce a binary probability vector"
            )
        probabilities = vector_to_array(column(probability_col))
        predicted = predicted.withColumn(
            "event_probability",
            F.when(
                F.size(probabilities) == 2, probabilities[int(self.event_label)]
            ).otherwise(F.raise_error("Expected a binary Spark probability vector")),
        )

        probability = "event_probability"
        if self.calibrator is not None:
            cal = LogisticCalibrator.from_dict(self.calibrator.to_dict())
            cal.output_probability = "event"
            temp = "__bmk_" + uuid4().hex
            if cal.input_space == "raw":
                value = vector_to_array(column(self.adapter.raw_col))[
                    int(self.event_label)
                ]
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
        return predicted.select(source + ".*", *[column(name) for name in names])
