"""Compose predictions on ordered original DataFrame inputs."""

from copy import deepcopy

import numpy as np
import pandas as pd

from brainmodelkit._probability import CapabilityError, choice, outputs
from brainmodelkit._probability_pandas import rounded
from brainmodelkit.calibration.pandas import LogisticCalibrator
from brainmodelkit.persistence import resolve_model
from brainmodelkit.scoring.pandas import ScoreAligner

from . import Workflow
from .adapters import OnnxClassifier, adapt, select_inputs, validated

TARGETS = (
    "raw",
    "event_probability",
    "calibrated_event_probability",
    "calibrated_non_event_probability",
    "business_score",
)


class PredictionPipeline(Workflow):
    """Compose a fitted binary classifier, optional calibration and fixed score.

    Model inputs are projected in declared feature order. Original features (including
    sklearn Pipeline preprocessing) are passed through the complete callable.
    Native LightGBM Boosters require explicit class_labels from training metadata.
    """

    def __init__(
        self,
        model,
        feature_cols: list[str] | None = None,
        *,
        event_label=None,
        class_labels=None,
        calibrator: LogisticCalibrator | None = None,
        score_aligner: ScoreAligner | None = None,
        prediction_callable=None,
        probability_output=None,
        raw_callable=None,
    ):
        if isinstance(model, PredictionPipeline):
            if (
                event_label is not None
                and event_label != model.event_label
                and (
                    model.prediction_callable is not None
                    or model.raw_callable is not None
                    or type(model.model).__module__.startswith("mlflow.pyfunc")
                )
            ):
                raise ValueError(
                    "The supplied callable output is bound to its original event_label"
                )
            event_label = model.event_label if event_label is None else event_label
            calibrator = model.calibrator if calibrator is None else calibrator
            score_aligner = (
                model.score_aligner if score_aligner is None else score_aligner
            )
            prediction_callable = prediction_callable or model.prediction_callable
            raw_callable = raw_callable or model.raw_callable
            probability_output = probability_output or model.probability_output
        if event_label is None:
            event_label = 1
        resolved = resolve_model(
            model, feature_cols=feature_cols, class_labels=class_labels
        )
        self.schema = resolved.schema
        self._inherit_metadata(model if isinstance(model, Workflow) else None)
        self.model = resolved.model
        self.feature_cols = resolved.feature_cols or []
        class_labels = resolved.class_labels
        if not self.feature_cols or len(set(self.feature_cols)) != len(
            self.feature_cols
        ):
            raise ValueError("feature_cols must be nonempty and unique")
        self.event_label = event_label
        self.calibrator = deepcopy(calibrator)
        self.score_aligner = deepcopy(score_aligner)
        self.prediction_callable = prediction_callable
        self.raw_callable = raw_callable
        if calibrator is not None:
            calibrator._ready()
            if calibrator.event_label != event_label:
                raise ValueError("Calibrator event_label differs from model event")
        for config in (calibrator, score_aligner):
            if (
                config is not None
                and config.feature_order
                and config.feature_order != self.feature_cols
            ):
                raise ValueError(
                    "Configuration feature_order differs from model inputs"
                )
        if class_labels is None and (
            prediction_callable is not None
            or raw_callable is not None
            or (
                probability_output is not None
                and type(self.model).__module__.startswith("mlflow.pyfunc")
            )
        ):
            self.classes = None
            self.event_index = None
        else:
            labels = getattr(self.model, "classes_", class_labels)
            if (
                labels is None
                or len(labels) != 2
                or len(set(labels)) != 2
                or event_label not in labels
            ):
                raise ValueError(
                    "Require two explicit model classes including event_label; "
                    "native Booster needs class_labels"
                )
            self.classes = list(labels)
            self.event_index = self.classes.index(event_label)

        predictor = self.model
        if (
            type(predictor).__module__.startswith("onnxruntime.")
            and prediction_callable is None
        ):
            predictor = OnnxClassifier(
                predictor, self.classes, probability_output=probability_output
            )
        if (
            type(predictor).__module__.startswith("mlflow.pyfunc")
            and prediction_callable is None
        ):
            if not isinstance(probability_output, str):
                raise ValueError("Select the pyfunc event probability_output column")

            def probability(frame):
                output = predictor.predict(frame)
                if (
                    not isinstance(output, pd.DataFrame)
                    or probability_output not in output
                ):
                    raise ValueError(
                        "pyfunc must return the declared event-probability column"
                    )
                return output[probability_output]

            prediction_callable = probability
        self.probability_output = probability_output
        self.adapter = adapt(
            predictor, self.event_index, prediction_callable, raw_callable
        )

    def predict(
        self, frame: pd.DataFrame, target: str = "event_probability"
    ) -> np.ndarray:
        choice(target, TARGETS, "target")
        frame = self._inputs(frame)
        if target == "raw":
            return self._raw(frame)
        if target == "event_probability":
            return validated(
                self.adapter.probability(frame), len(frame), probability=True
            )
        if self.calibrator is None and target.startswith("calibrated"):
            raise ValueError("This target requires a calibrator")
        p = self._calibrated(frame)
        if target == "calibrated_non_event_probability":
            return 1 - p
        if target == "business_score":
            if self.score_aligner is None:
                raise ValueError("business_score requires a ScoreAligner")
            # Composition has known event semantics; display choices never flip scoring.
            return self.score_aligner.score_event(p)
        return p

    def _raw(self, frame):
        raw = getattr(self.adapter, "raw", None)
        if raw is None:
            raise CapabilityError(
                "This adapter does not provide event-oriented raw output"
            )
        return validated(raw(frame), len(frame))

    def _inputs(self, frame):
        selected = select_inputs(frame, self.feature_cols)
        if self.schema is not None:
            for name, dtype in self.schema.items():
                if pd.api.types.is_numeric_dtype(
                    pd.api.types.pandas_dtype(dtype)
                ) and not pd.api.types.is_numeric_dtype(selected[name]):
                    raise TypeError(f"Expected numeric inputs for {name!r}")
        return selected

    def _calibrated(self, frame, p=None):
        cal = self.calibrator
        if cal is not None and cal.input_space == "raw":
            return cal.predict_event(self._raw(frame))
        if p is None:
            p = validated(self.adapter.probability(frame), len(frame), probability=True)
        if cal is None:
            return p
        return cal.predict_event(1 - p if cal.input_probability == "non_event" else p)

    def transform(self, df: pd.DataFrame, *, overwrite: bool = False) -> pd.DataFrame:
        """Retain source columns and add explicitly named workflow outputs."""
        names = ["event_probability"]
        if self.calibrator is not None:
            names += [
                "calibrated_event_probability",
                "calibrated_non_event_probability",
            ]
        if self.score_aligner is not None:
            names += ["business_score_unrounded", "business_score"]
        outputs(df.columns, names, overwrite)
        features = self._inputs(df)
        result = df.copy()
        p = validated(
            self.adapter.probability(features), len(features), probability=True
        )
        result["event_probability"] = p
        if self.calibrator is not None:
            p = self._calibrated(features, p)
            result["calibrated_event_probability"] = p
            result["calibrated_non_event_probability"] = 1 - p
        if self.score_aligner is not None:
            align = self.score_aligner
            points = align.score_event(p)
            result["business_score_unrounded"] = points
            result["business_score"] = np.clip(
                rounded(points, align.decimals), align.score_min, align.score_max
            )
        return result

    def metadata(self) -> dict:
        return {
            "model_class": type(self.model).__module__
            + "."
            + type(self.model).__name__,
            "feature_order": self.feature_cols,
            "event_label": self.event_label,
            "calibration": self.calibrator.to_dict()
            if self.calibrator is not None
            else None,
            "scoring": self.score_aligner.to_dict()
            if self.score_aligner is not None
            else None,
        }
