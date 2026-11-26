"""SHAP adapters for numeric original features and complete prediction functions."""

import hashlib
from copy import deepcopy
from importlib.metadata import version

import numpy as np
import pandas as pd

from brainmodelkit._probability import CapabilityError, choice
from brainmodelkit._probability_pandas import numeric, rounded
from brainmodelkit.prediction.pandas import TARGETS, PredictionPipeline, select_inputs

from ._result import ExplanationResult


def _features(frame, names, dtypes=None):
    if list(frame.columns) != names:
        raise ValueError(
            f"Feature columns/order must be exactly {names}; select features explicitly"
        )

    if dtypes is not None and [str(frame[name].dtype) for name in names] != dtypes:
        raise TypeError(
            "Feature dtypes differ from reference; cast explicitly before explaining"
        )
    return frame


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
        event_label=None,
        class_labels=None,
        calibrator=None,
        score_aligner=None,
        prediction_callable=None,
        probability_output=None,
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
        if isinstance(model, PredictionPipeline):
            if feature_cols is not None and list(feature_cols) != model.feature_cols:
                raise ValueError("Conflicting feature order")
            if event_label is not None and model.event_label != event_label:
                raise ValueError("Conflicting event_label on PredictionPipeline")
        self.pipeline = (
            model
            if isinstance(model, PredictionPipeline)
            else PredictionPipeline(
                model,
                feature_cols,
                event_label=event_label,
                class_labels=class_labels,
                calibrator=calibrator,
                score_aligner=score_aligner,
                prediction_callable=prediction_callable,
                probability_output=probability_output,
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
                probability_output,
                class_labels,
            )
        ):
            raise ValueError("Configure components on PredictionPipeline, not twice")
        background = select_inputs(background, self.features)
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
        self.feature_dictionary = deepcopy(
            feature_dictionary or getattr(self.pipeline, "feature_dictionary", {})
        )
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
        # Independent masking operates on a numeric matrix. For discrete or
        # nullable inputs, mask row indices and recover the original pandas values
        # before prediction. This preserves None, categorical types and large ints.
        self._lookup = {}
        if method == "permutation":
            for name, dtype in self.background.dtypes.items():
                if not pd.api.types.is_float_dtype(dtype):
                    self._lookup[name] = self.background[name].reset_index(drop=True)
        else:
            for name in self.features:
                numeric(self.background[name])
        self._affine = (0.0, 1.0)
        # Fail unsupported targets before any expensive explanation work.
        self.pipeline.predict(self.background.iloc[:1], target)
        if method == "permutation" and max_evals < 2 * len(self.features) + 1:
            raise ValueError("max_evals must be at least 2 * feature_count + 1")

    def _encoded(self, frame, offset=0):
        if self.method != "permutation":
            return frame
        encoded = frame.copy()
        for name in self._lookup:
            encoded[name] = np.arange(offset, offset + len(frame))
        return encoded.astype(float)

    def _initialize(self):
        try:
            import shap
        except ImportError as exc:
            raise ImportError("Install brainmodelkit[explainability] for SHAP") from exc
        masker = shap.maskers.Independent(
            self._encoded(self.background), max_samples=len(self.background)
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
                frame = pd.DataFrame(values, columns=self.features)
                for name, lookup in self._lookup.items():
                    frame[name] = lookup.iloc[frame[name].to_numpy(dtype=int)].array
                frame = frame.astype(self.background.dtypes.to_dict())
                return pipeline.predict(frame, explained)

            self._explainer = shap.PermutationExplainer(predict, masker, seed=self.seed)
            return
        model = pipeline.model
        kind = getattr(pipeline.adapter, "optimized_kind", None)
        if target.startswith("calibrated") or (
            target == "business_score" and pipeline.calibrator is not None
        ):
            raise CapabilityError(
                "Nonlinear calibration requires explicit method='permutation' "
                "for the complete prediction function"
            )
        if kind == "linear_margin" and target == "raw":
            self._explainer = shap.LinearExplainer(model, masker)
            self._affine = (0.0, 1.0 if pipeline.event_index == 1 else -1.0)
            return
        if kind in {"class_probability_tree", "boosted_tree"}:
            self._multi_output = kind == "class_probability_tree"
            self._explainer = shap.TreeExplainer(
                model,
                data=masker,
                feature_perturbation="interventional",
                model_output="raw" if target == "raw" else "probability",
            )
            if not self._multi_output and pipeline.event_index == 0:
                self._affine = (0.0 if target == "raw" else 1.0, -1.0)
            if target == "business_score":
                offset, factor = pipeline.score_aligner.affine
                a, b = self._affine
                self._affine = (offset + factor * a, factor * b)
            return
        if kind == "lightgbm":
            from lightgbm import Booster

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
            "No optimized adapter for this model and target. Use method='permutation' "
            "for sklearn preprocessing pipelines or other compatible callables."
        )

    def explain(self, df: pd.DataFrame) -> ExplanationResult:
        if len(df) > self.max_rows:
            raise ValueError(
                "Explanation exceeds max_rows; select an explicit bounded batch"
            )
        df = select_inputs(df, self.features + self.keys)
        frame = _features(df[self.features], self.features, self.dtypes)
        if self.keys and (
            df[self.keys].isna().any().any() or df.duplicated(self.keys).any()
        ):
            raise ValueError(
                "Observation keys must be non-null and unique within the batch"
            )
        if frame.empty:
            raise ValueError("Explanation batch must not be empty")
        for name in self._lookup:
            if frame[name].dtype != self.background[name].dtype:
                raise TypeError(f"Feature dtypes differ from reference for {name!r}")
            self._lookup[name] = pd.concat(
                [self.background[name], frame[name]], ignore_index=True
            )
        if self._explainer is None:
            self._initialize()
        explanation = (
            self._explainer(
                self._encoded(frame, len(self.background)),
                max_evals=self.max_evals,
                silent=True,
            )
            if self.method == "permutation"
            else self._explainer(frame)
        )
        values = np.asarray(explanation.values, dtype=float)
        base_values = explanation.base_values
        if getattr(self, "_multi_output", False):
            values = values[:, :, self.pipeline.event_index]
            base_values = np.asarray(base_values)[:, self.pipeline.event_index]
        bases = np.broadcast_to(
            np.asarray(base_values, dtype=float), (len(frame),)
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
            getattr(self.pipeline.adapter, "raw_units", "adapter-defined raw output")
            if self.target == "raw"
            else "score points"
            if self.target == "business_score"
            else "probability (0 to 1)"
        )
        metadata = {
            **self.pipeline.metadata(),
            "target": self.target,
            "units": units,
            "feature_space": getattr(self.pipeline, "feature_space", "original"),
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
