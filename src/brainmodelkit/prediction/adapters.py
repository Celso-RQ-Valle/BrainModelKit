"""Binary prediction adapters. Registration is explicit and process-local."""

import json

import numpy as np

from brainmodelkit._probability import CapabilityError
from brainmodelkit._probability_pandas import numeric

_factories = {}


def register_adapter(model_type, factory):
    """Register factory(model, event_index) returning probability/raw methods."""
    if not isinstance(model_type, type) or not callable(factory):
        raise TypeError("Provide a model class and an adapter factory")
    _factories[model_type] = factory


def select_inputs(frame, names):
    if len(frame.columns) != len(set(frame.columns)):
        raise ValueError("Duplicate input columns")
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    return frame.loc[:, names]


class OnnxClassifier:
    """Map a named ONNX probability output to an ordered binary matrix."""

    def __init__(self, session, class_labels, *, probability_output=None):
        self.session = session
        self.classes_ = list(class_labels)
        outputs = [o.name for o in session.get_outputs()]
        candidates = [
            n for n in outputs if n in {"probabilities", "output_probability"}
        ]
        self.output = probability_output or (
            candidates[0] if len(candidates) == 1 else None
        )
        if self.output not in outputs:
            raise ValueError("Supply probability_output for this ONNX session")

    def predict_proba(self, frame):
        inputs = self.session.get_inputs()
        if len(inputs) > 1 and {inp.name for inp in inputs} != set(frame.columns):
            raise ValueError("Named ONNX inputs must match the declared features")
        dtypes = {
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(string)": str,
            "tensor(bool)": bool,
        }
        feed = {}
        for inp in inputs:
            if inp.type not in dtypes:
                raise CapabilityError(f"Unsupported ONNX input type {inp.type}")
            if (
                len(inputs) == 1
                and len(inp.shape) == 2
                and (
                    not isinstance(inp.shape[1], int)
                    or inp.shape[1] == len(frame.columns)
                )
            ):
                data = frame
            elif inp.name in frame and (len(inputs) > 1 or len(frame.columns) == 1):
                data = frame[[inp.name]]
            else:
                raise ValueError(f"Cannot map ONNX input {inp.name} to features")
            if inp.type in {"tensor(int64)", "tensor(int32)"}:
                from pandas.api.types import is_integer_dtype

                if not all(is_integer_dtype(dtype) for dtype in data.dtypes):
                    raise TypeError(f"ONNX input {inp.name} requires integer features")
                limits = np.iinfo(dtypes[inp.type])
                if (
                    data.isna().any().any()
                    or ((data < limits.min) | (data > limits.max)).any().any()
                ):
                    raise ValueError(f"ONNX input {inp.name} exceeds its integer dtype")
            array = data.to_numpy(dtype=dtypes[inp.type])
            feed[inp.name] = array.ravel() if len(inp.shape) == 1 else array
        output = self.session.run([self.output], feed)[0]
        if isinstance(output, list) and output and isinstance(output[0], dict):
            if any(set(row) != set(self.classes_) for row in output):
                raise ValueError(
                    "ONNX probability map keys differ from binary class_labels"
                )
            return np.asarray(
                [[row[label] for label in self.classes_] for row in output]
            )
        return np.asarray(output)


class BinaryAdapter:
    def __init__(self, model, event_index, probability=None, raw=None):
        self.model, self.index = model, event_index
        self.probability_fn, self.raw_fn = probability, raw
        self.framework = type(model).__module__.split(".")[0]
        if self.framework == "xgboost" and probability is None:
            booster = model.get_booster() if hasattr(model, "get_booster") else model
            objective = json.loads(booster.save_config())["learner"]["objective"][
                "name"
            ]
            if objective != "binary:logistic":
                raise CapabilityError(
                    "Expected binary:logistic or an explicit probability callable"
                )

    @property
    def optimized_kind(self):
        """Eligible uncalibrated model structure; evaluated only on request."""
        if self.probability_fn is not None or self.raw_fn is not None:
            return None
        if self.framework == "lightgbm":
            from lightgbm import Booster, LGBMClassifier

            return (
                "lightgbm"
                if isinstance(self.model, (Booster, LGBMClassifier))
                else None
            )
        if self.framework == "sklearn":
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                RandomForestClassifier,
            )
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier

            if isinstance(self.model, LogisticRegression):
                return "linear_margin"
            if isinstance(self.model, (RandomForestClassifier, DecisionTreeClassifier)):
                return "class_probability_tree"
            if isinstance(self.model, GradientBoostingClassifier):
                return "boosted_tree"
        return None

    @property
    def raw_units(self):
        if self.raw_fn is not None:
            return "caller-defined event-oriented raw output"
        if (
            self.optimized_kind == "linear_margin"
            and getattr(self.model, "multi_class", None) != "multinomial"
        ):
            return "event log-odds"
        return "event-oriented model margin"

    def _native(self, frame, raw):
        model = self.model
        if self.framework == "lightgbm":
            booster = getattr(model, "booster_", model)
            if str(booster.dump_model().get("objective", "")).split()[0] != "binary":
                raise CapabilityError("Expected a binary LightGBM objective")
            values = booster.predict(frame, raw_score=raw)
        elif self.framework == "xgboost":
            import xgboost as xgb

            if isinstance(model, xgb.Booster):
                best = model.attr("best_iteration")
                rounds = (0, int(best) + 1) if best is not None else (0, 0)
                values = model.predict(
                    xgb.DMatrix(
                        frame, enable_categorical="c" in (model.feature_types or [])
                    ),
                    output_margin=raw,
                    iteration_range=rounds,
                )
            else:
                values = model.predict(frame, output_margin=raw)
        elif self.framework == "catboost" and raw:
            values = model.predict(frame, prediction_type="RawFormulaVal")
        else:
            raise CapabilityError(
                "Requested output unavailable; supply an explicit callable"
            )
        if self.index == 0:
            return -values if raw else 1 - values
        return values

    def probability(self, frame):
        if self.probability_fn is not None:
            return self.probability_fn(frame)
        if hasattr(self.model, "predict_proba"):
            matrix = np.asarray(self.model.predict_proba(frame))
            if matrix.shape != (len(frame), 2):
                raise ValueError(
                    "Expected a binary probability matrix with two columns"
                )
            return matrix[:, self.index]
        return self._native(frame, False)

    def raw(self, frame):
        if self.raw_fn is not None:
            return self.raw_fn(frame)
        if hasattr(self.model, "decision_function"):
            values = self.model.decision_function(frame)
            return -values if self.index == 0 else values
        return self._native(frame, True)


def adapt(model, index, probability, raw):
    if probability is None and raw is None:
        for cls in type(model).__mro__:
            if cls in _factories:
                return _factories[cls](model, index)
    return BinaryAdapter(model, index, probability, raw)


def validated(values, rows, *, probability=False):
    if np.shape(values) != (rows,):
        raise ValueError("Expected one event-oriented value per row")
    return numeric(values, bounds=(0, 1) if probability else None)
