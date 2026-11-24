"""SHAP adapters for numeric original features and complete prediction functions."""

import hashlib
from copy import deepcopy
from importlib.metadata import version

import numpy as np
import pandas as pd

from brainmodelkit._probability import CapabilityError, choice, outputs
from brainmodelkit._probability_pandas import numeric, rounded
from brainmodelkit.calibration.pandas import LogisticCalibrator
from brainmodelkit.scoring.pandas import ScoreAligner
from brainmodelkit.training import TrainingResult

from ._result import ExplanationResult

TARGETS = (
    "raw",
    "event_probability",
    "calibrated_event_probability",
    "calibrated_non_event_probability",
    "business_score",
)


def _features(frame, names, dtypes=None):
    if list(frame.columns) != names:
        raise ValueError(
            f"Feature columns/order must be exactly {names}; select features explicitly"
        )
    for name in names:
        numeric(frame[name])
    if dtypes is not None and [str(frame[name].dtype) for name in names] != dtypes:
        raise TypeError(
            "Feature dtypes differ from reference; cast explicitly before explaining"
        )
    return frame


def _unwrap(model):
    return model.model if isinstance(model, TrainingResult) else model


class PredictionPipeline:
    """Compose a fitted binary classifier, optional calibration and fixed score.

    Feature frames must match feature_cols exactly. Original features (including
    sklearn Pipeline preprocessing) are passed through the complete callable.
    Native LightGBM Boosters require explicit class_labels from training metadata.
    """

    def __init__(
        self,
        model,
        feature_cols: list[str],
        *,
        event_label=1,
        class_labels=None,
        calibrator: LogisticCalibrator | None = None,
        score_aligner: ScoreAligner | None = None,
        prediction_callable=None,
        raw_callable=None,
    ):
        self.model = _unwrap(model)
        self.feature_cols = list(feature_cols)
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
        model_features = getattr(self.model, "feature_names_in_", None)
        if model_features is not None and list(model_features) != self.feature_cols:
            raise ValueError("feature_cols differ from fitted model feature_names_in_")
        if prediction_callable is not None:
            self.classes = None
            self.event_index = None
        else:
            labels = getattr(self.model, "classes_", class_labels)
            if labels is None or len(labels) != 2 or event_label not in labels:
                raise ValueError(
                    "Require two explicit model classes including event_label; "
                    "native Booster needs class_labels"
                )
            self.classes = list(labels)
            self.event_index = self.classes.index(event_label)

    def _base_event(self, frame):
        if self.prediction_callable is not None:
            values = self.prediction_callable(frame)
        elif hasattr(self.model, "predict_proba"):
            values = self.model.predict_proba(frame)[:, self.event_index]
        else:
            from lightgbm import Booster

            if (
                not isinstance(self.model, Booster)
                or self.model.num_model_per_iteration() != 1
            ):
                raise CapabilityError(
                    "Need binary predict_proba or an explicit "
                    "event-probability callable"
                )
            values = self.model.predict(frame)
            if self.event_index == 0:
                values = 1 - values
        result = numeric(values, bounds=(0, 1))
        if result.shape != (len(frame),):
            raise ValueError(
                "Prediction callable must return one event probability per row"
            )
        return result

    def _raw(self, frame):
        if self.raw_callable is not None:
            values = self.raw_callable(frame)
        elif hasattr(self.model, "decision_function"):
            values = self.model.decision_function(frame)
            if self.event_index == 0:
                values = -values
        else:
            from lightgbm import Booster, LGBMClassifier

            if not isinstance(self.model, (Booster, LGBMClassifier)):
                raise CapabilityError(
                    "Raw output unavailable; provide raw_callable explicitly"
                )
            values = self.model.predict(frame, raw_score=True)
            if self.event_index == 0:
                values = -values
        result = numeric(values)
        if result.shape != (len(frame),):
            raise ValueError(
                "Raw callable must return one event-oriented margin per row"
            )
        return result

    def predict(
        self, frame: pd.DataFrame, target: str = "event_probability"
    ) -> np.ndarray:
        choice(target, TARGETS, "target")
        _features(frame, self.feature_cols)
        if target == "raw":
            return self._raw(frame)
        p = self._base_event(frame)
        if target == "event_probability":
            return p
        if self.calibrator is not None:
            cal = self.calibrator
            values = (
                self._raw(frame)
                if cal.input_space == "raw"
                else 1 - p
                if cal.input_probability == "non_event"
                else p
            )
            p = cal.predict_event(values)
        elif target.startswith("calibrated"):
            raise ValueError("This target requires a calibrator")
        if target == "calibrated_non_event_probability":
            return 1 - p
        if target == "business_score":
            if self.score_aligner is None:
                raise ValueError("business_score requires a ScoreAligner")
            # Composition has known event semantics; display choices never flip scoring.
            return self.score_aligner.score_event(p)
        return p

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
        features = df[self.feature_cols]
        result = df.copy()
        for name in names:
            if name.startswith("business_score"):
                values = self.predict(features, "business_score")
                if name == "business_score":
                    values = np.clip(
                        rounded(values, self.score_aligner.decimals),
                        self.score_aligner.score_min,
                        self.score_aligner.score_max,
                    )
            else:
                values = self.predict(features, name)
            result[name] = values
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


class ModelExplainer:
    """Explicit optimized SHAP or bounded permutation SHAP of the full callable.

    Optimized: logistic raw margins; binary LightGBM raw/probability.
    Pipelines and nonlinear calibrated outputs require method='permutation'.
    Background is required, sampled once with a fixed seed, then retained.
    """

    def __init__(
        self,
        model,
        background: pd.DataFrame,
        feature_cols: list[str] | None = None,
        *,
        target: str = "raw",
        method: str = "optimized",
        event_label=1,
        class_labels=None,
        calibrator=None,
        score_aligner=None,
        prediction_callable=None,
        raw_callable=None,
        key_cols: list[str] | None = None,
        max_background: int = 100,
        max_rows: int = 1000,
        max_evals: int = 501,
        seed: int = 42,
        reference_id: str | None = None,
        feature_dictionary: dict | None = None,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ):
        choice(target, TARGETS, "target")
        choice(method, ("optimized", "permutation"), "method")
        if any(
            type(value) is not int or value < 1
            for value in (max_background, max_rows, max_evals)
        ):
            raise ValueError(
                "Background, row, and evaluation limits must be positive integers"
            )
        if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
            raise ValueError(
                "Reconstruction tolerances must be finite and non-negative"
            )
        self.pipeline = (
            model
            if isinstance(model, PredictionPipeline)
            else PredictionPipeline(
                model,
                feature_cols or list(background.columns),
                event_label=event_label,
                class_labels=class_labels,
                calibrator=calibrator,
                score_aligner=score_aligner,
                prediction_callable=prediction_callable,
                raw_callable=raw_callable,
            )
        )
        self.features = self.pipeline.feature_cols
        if isinstance(model, PredictionPipeline) and any(
            value is not None
            for value in (
                calibrator,
                score_aligner,
                prediction_callable,
                raw_callable,
                class_labels,
            )
        ):
            raise ValueError("Configure components on PredictionPipeline, not twice")
        _features(background, self.features)
        if background.empty:
            raise ValueError("Background must contain observations")
        self.background = background.sample(
            n=min(len(background), max_background), random_state=seed
        ).copy()
        self.dtypes = [str(self.background[name].dtype) for name in self.features]
        self.keys = list(key_cols or [])
        if len(set(self.keys)) != len(self.keys) or set(self.keys) & set(self.features):
            raise ValueError("key_cols must be unique and separate from features")
        reserved = {
            "prediction",
            "base_value",
            "residual",
            "tolerance",
            "reconstruction_ok",
            "rounded_prediction",
            "rounding_difference",
        }
        if set(self.keys) & reserved or any(
            name.startswith(("contribution::", "value::")) for name in self.keys
        ):
            raise ValueError("Key column collides with explanation outputs")
        if set(feature_dictionary or {}) - set(self.features):
            raise ValueError("Feature dictionary includes unknown features")
        self.target, self.method = target, method
        self.max_rows, self.max_evals, self.seed = max_rows, max_evals, seed
        self.atol, self.rtol = atol, rtol
        self.feature_dictionary = deepcopy(feature_dictionary or {})
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(self.background, index=True).values.tobytes()
        ).hexdigest()
        self.reference = {
            "id": reference_id,
            "sha256": digest,
            "rows": len(self.background),
            "seed": seed,
            "dtypes": self.dtypes,
            "feature_order": self.features,
        }
        self._explainer = None
        self._affine = (0.0, 1.0)
        # Fail unsupported targets before any expensive explanation work.
        self.pipeline.predict(self.background.iloc[:1], target)
        if method == "permutation" and max_evals < 2 * len(self.features) + 1:
            raise ValueError("max_evals must be at least 2 * feature_count + 1")

    def _initialize(self):
        try:
            import shap
        except ImportError as exc:
            raise ImportError("Install brainmodelkit[explainability] for SHAP") from exc
        masker = shap.maskers.Independent(
            self.background, max_samples=len(self.background)
        )
        pipeline = self.pipeline
        target = self.target
        if self.method == "permutation":
            # Explain final event probability then apply only exact affine display maps.
            explained = target
            if target == "calibrated_non_event_probability":
                explained = "calibrated_event_probability"
                self._affine = (1.0, -1.0)
            elif target == "business_score":
                explained = (
                    "calibrated_event_probability"
                    if pipeline.calibrator
                    else "event_probability"
                )
                self._affine = pipeline.score_aligner.affine

            def predict(values):
                frame = pd.DataFrame(values, columns=self.features).astype(
                    dict(zip(self.features, self.dtypes, strict=True))
                )
                return pipeline.predict(frame, explained)

            self._explainer = shap.PermutationExplainer(predict, masker, seed=self.seed)
            return
        from sklearn.linear_model import LogisticRegression

        model = pipeline.model
        if (
            pipeline.prediction_callable is not None
            or pipeline.raw_callable is not None
        ):
            raise CapabilityError("Custom callables require method='permutation'")
        if target.startswith("calibrated") or (
            target == "business_score" and pipeline.calibrator is not None
        ):
            raise CapabilityError(
                "Nonlinear calibration requires explicit method='permutation' "
                "for the complete prediction function"
            )
        if isinstance(model, LogisticRegression) and target == "raw":
            self._explainer = shap.LinearExplainer(model, masker)
            self._affine = (0.0, 1.0 if pipeline.event_index == 1 else -1.0)
            return
        from lightgbm import Booster, LGBMClassifier

        if isinstance(model, (Booster, LGBMClassifier)):
            if isinstance(model, Booster):
                objective = str(model.dump_model().get("objective", ""))
                if not objective.startswith("binary"):
                    raise CapabilityError("Native Booster must use a binary objective")
                # A loaded native Booster loses params. Restore objective metadata
                # on a private copy so SHAP does not mistake it for regression.
                model = Booster(model_str=model.model_to_string())
                model.params["objective"] = "binary"
            output = "raw" if target == "raw" else "probability"
            self._explainer = shap.TreeExplainer(
                model,
                data=masker,
                feature_perturbation="interventional",
                model_output=output,
            )
            if pipeline.event_index == 0:
                self._affine = (0.0 if target == "raw" else 1.0, -1.0)
            if target == "business_score":
                offset, factor = pipeline.score_aligner.affine
                base_offset, base_factor = self._affine
                self._affine = (offset + factor * base_offset, factor * base_factor)
            return
        raise CapabilityError(
            "Optimized support is logistic raw output or binary LightGBM "
            "raw/probability on numeric model inputs. Use method='permutation' "
            "for sklearn preprocessing pipelines or other compatible callables."
        )

    def explain(self, df: pd.DataFrame) -> ExplanationResult:
        if len(df) > self.max_rows:
            raise ValueError(
                "Explanation exceeds max_rows; select an explicit bounded batch"
            )
        if len(df.columns) != len(set(df.columns)) or set(df.columns) != set(
            self.features + self.keys
        ):
            raise ValueError(
                "Explain frame must contain exactly features and configured keys"
            )
        if [name for name in df.columns if name not in self.keys] != self.features:
            raise ValueError("Feature order differs from fitted schema")
        frame = _features(df[self.features], self.features, self.dtypes)
        if self.keys and (
            df[self.keys].isna().any().any() or df.duplicated(self.keys).any()
        ):
            raise ValueError(
                "Observation keys must be non-null and unique within the batch"
            )
        if frame.empty:
            raise ValueError("Explanation batch must not be empty")
        if self._explainer is None:
            self._initialize()
        explanation = (
            self._explainer(frame, max_evals=self.max_evals, silent=True)
            if self.method == "permutation"
            else self._explainer(frame)
        )
        values = np.asarray(explanation.values, dtype=float)
        bases = np.broadcast_to(
            np.asarray(explanation.base_values, dtype=float), (len(frame),)
        ).copy()
        if values.shape != frame.shape:
            raise CapabilityError(
                "Explainer returned unsupported multiclass/multioutput attributions"
            )
        offset, factor = self._affine
        values, bases = factor * values, offset + factor * bases
        predictions = self.pipeline.predict(frame, self.target)
        residuals = predictions - (bases + values.sum(axis=1))
        tolerances = self.atol + self.rtol * np.abs(predictions)
        if not np.isfinite(residuals).all() or (np.abs(residuals) > tolerances).any():
            raise RuntimeError(
                "Explanation reconstruction failed in the declared output space"
            )
        units = (
            "raw margin (log-odds for logistic/LightGBM)"
            if self.target == "raw"
            else "score points"
            if self.target == "business_score"
            else "probability (0 to 1)"
        )
        metadata = {
            **self.pipeline.metadata(),
            "target": self.target,
            "units": units,
            "feature_space": "original numeric inputs",
            "reference": self.reference,
            "method": self.method,
            "max_evals": self.max_evals if self.method == "permutation" else None,
            "seed": self.seed,
            "feature_dictionary": self.feature_dictionary,
            "atol": self.atol,
            "rtol": self.rtol,
            "versions": {name: version(name) for name in ("shap", "numpy", "pandas")},
            "execution": "local",
            "approximation": "permutation sampling"
            if self.method == "permutation"
            else "optimized interventional SHAP",
        }
        display = None
        if (
            self.target == "business_score"
            and self.pipeline.score_aligner.decimals is not None
        ):
            align = self.pipeline.score_aligner
            display = np.clip(
                rounded(predictions, align.decimals), align.score_min, align.score_max
            )
        return ExplanationResult(
            predictions,
            bases,
            pd.DataFrame(values, columns=self.features, index=df.index),
            frame.copy(),
            df[self.keys].copy(),
            metadata,
            residuals,
            tolerances,
            display,
        )
