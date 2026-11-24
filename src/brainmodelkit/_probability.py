"""Backend-independent contracts for probability postprocessing."""

import json
import math
from dataclasses import asdict, dataclass, field
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from brainmodelkit import __version__


class CapabilityError(NotImplementedError):
    """The selected model/runtime cannot implement the requested operation."""


def choice(value, allowed, name):
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}")


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def outputs(columns, names, overwrite=False):
    names = [name for name in names if name is not None]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Output names must be non-empty strings")
    if len(set(columns)) != len(columns) or len(set(names)) != len(names):
        raise ValueError("Duplicate input or output columns")
    if not overwrite and set(columns).intersection(names):
        raise ValueError("Output column collision; use overwrite=True explicitly")


class JsonConfig:
    """Versioned JSON configuration, portable between dataframe backends."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": self.__class__.__name__,
            "brainmodelkit_version": __version__,
            "config": asdict(self),
        }

    @classmethod
    def from_dict(cls, data: dict):
        if data.get("schema_version") != 1 or data.get("kind") != cls.__name__:
            raise ValueError("Unsupported configuration version or kind")
        return cls(**data["config"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class CalibrationConfig(JsonConfig):
    intercept: float | None = None
    slope: float | None = None
    input_space: str = "logit"
    input_probability: str | None = None
    output_probability: str = "event"
    event_label: Any = 1
    epsilon: float = 1e-12
    invalid_policy: str = "raise"
    mode: str | None = None
    diagnostics_: dict = field(default_factory=dict)
    feature_order: list[str] = field(default_factory=list)

    def __post_init__(self):
        choice(self.input_space, ("logit", "probability", "raw"), "input_space")
        choice(self.invalid_policy, ("raise", "null"), "invalid_policy")
        choice(self.output_probability, ("event", "non_event"), "output_probability")
        if self.input_space == "raw":
            if self.input_probability is not None:
                raise ValueError("Raw margins require input_probability=None")
        else:
            if self.input_probability is None:
                self.input_probability = "event"
            choice(self.input_probability, ("event", "non_event"), "input_probability")
        finite(self.epsilon, "epsilon")
        if not 0 < self.epsilon < 0.5:
            raise ValueError("epsilon must be between 0 and 0.5 exclusively")
        if not isinstance(self.event_label, (str, bool, int, float)):
            raise ValueError(
                "event_label must be a JSON scalar string, boolean or number"
            )
        json.dumps(self.event_label, allow_nan=False)
        if (self.intercept is None) != (self.slope is None):
            raise ValueError("Supply both intercept and slope, or neither for fitting")
        if self.intercept is not None:
            finite(self.intercept, "intercept")
            finite(self.slope, "slope")
        if self.mode is None:
            self.mode = "manual" if self.intercept is not None else "unfitted"
        choice(self.mode, ("manual", "fitted", "unfitted"), "mode")
        if (self.mode == "unfitted") != (self.intercept is None):
            raise ValueError("Mode and parameter availability disagree")

    def _ready(self):
        if self.intercept is None:
            raise ValueError("Fit the calibrator or supply manual intercept and slope")

    @property
    def ranking(self) -> str:
        self._ready()
        if self.input_space == "raw":
            return "model_defined"
        return (
            "preserves"
            if self.slope > 0
            else "reverses"
            if self.slope < 0
            else "collapses"
        )

    def _fit_options(self, max_iter, tol):
        if self.mode == "manual":
            raise ValueError(
                "Manual parameters cannot be silently refitted; create a new calibrator"
            )
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        finite(tol, "tol")
        if tol <= 0:
            raise ValueError("tol must be positive")

    def _fitted(self, intercept, slope, diagnostics):
        finite(intercept, "intercept")
        finite(slope, "slope")
        self.intercept, self.slope, self.mode = intercept, slope, "fitted"
        self.diagnostics_ = diagnostics
        return self


@dataclass
class ScoringConfig(JsonConfig):
    score_min: float = 0.0
    score_max: float = 1000.0
    input_probability: str = "event"
    higher_score_means: str = "lower_event_probability"
    decimals: int | None = None
    invalid_policy: str = "raise"
    feature_order: list[str] = field(default_factory=list)

    def __post_init__(self):
        finite(self.score_min, "score_min")
        finite(self.score_max, "score_max")
        if self.score_min >= self.score_max or not math.isfinite(
            self.score_max - self.score_min
        ):
            raise ValueError("Require finite score_min < score_max and finite range")
        choice(self.input_probability, ("event", "non_event"), "input_probability")
        choice(
            self.higher_score_means,
            ("lower_event_probability", "higher_event_probability"),
            "higher_score_means",
        )
        choice(self.invalid_policy, ("raise", "null"), "invalid_policy")
        if self.decimals is not None and (
            type(self.decimals) is not int or not 0 <= self.decimals <= 12
        ):
            raise ValueError(
                "decimals must be None or an integer from 0 to 12 (half-even)"
            )
        if self.decimals is not None:
            with localcontext() as context:
                context.prec = 340
                quantum = Decimal(1).scaleb(-self.decimals)
                for bound in (self.score_min, self.score_max):
                    value = Decimal(str(bound))
                    if value != value.quantize(quantum):
                        raise ValueError(
                            "Score bounds must be representable at the requested decimals"
                        )

    @property
    def affine(self) -> tuple[float, float]:
        width = self.score_max - self.score_min
        if self.higher_score_means == "lower_event_probability":
            return self.score_max, -width
        return self.score_min, width


def check_fit_summary(n, lo0, hi0, lo1, hi1):
    if n < 3 or any(v is None for v in (lo0, hi0, lo1, hi1)):
        raise ValueError(
            "Need at least three valid positive-weight observations "
            "and both target classes"
        )
    if min(lo0, lo1) == max(hi0, hi1):
        raise ValueError("Constant calibration predictor: slope is not identifiable")
    if hi0 <= lo1 or hi1 <= lo0:
        raise ValueError(
            "Separated calibration data: unpenalized finite coefficients "
            "are not identifiable"
        )
