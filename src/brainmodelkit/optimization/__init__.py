"""Optuna-based model optimization for supported dataframe backends."""

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class OptimizationResult:
    """Study, selected parameters, diagnostics, and the final full-data model."""

    study: Any
    best_params: dict[str, Any]
    best_value: float
    trials: Any
    best_model: Any
    selected_trial_number: int | None = None
    constraint_satisfied: bool | None = None
    trials_path: str | None = None


def require_optuna():
    """Import Optuna only when optimization is requested."""
    try:
        import optuna
    except ImportError as error:
        raise ImportError(
            "Optuna is required for optimization. Install "
            "`brainmodelkit[optimization]`."
        ) from error
    return optuna


def normalize_search_space(search_space: dict[str, Any]) -> dict[str, Any]:
    """Translate constants, boolean choices, and numeric intervals."""
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("search_space must be a non-empty dictionary")
    normalized = {}
    for name, specification in search_space.items():
        if not isinstance(name, str) or not name:
            raise ValueError("search_space parameter names must be non-empty strings")
        if isinstance(specification, list):
            if len(specification) == 1:
                value = specification[0]
                if (
                    value is None
                    or isinstance(value, (str, bool))
                    or (isinstance(value, (int, float)) and math.isfinite(value))
                ):
                    normalized[name] = {"type": "categorical", "choices": [value]}
                    continue
            if len(specification) == 2 and all(
                isinstance(value, bool) for value in specification
            ):
                normalized[name] = {"type": "categorical", "choices": specification}
                continue
            if len(specification) != 2 or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in specification
            ):
                raise ValueError(
                    f"Interval for {name!r} must be [minimum, maximum] with two "
                    "finite numeric bounds, [False, True] boolean choices, "
                    "or [value] with one finite scalar; "
                    "use a categorical dictionary for choices"
                )
            low, high = specification
            if low > high:
                raise ValueError(f"Interval for {name!r} requires minimum <= maximum")
            specification = {
                "type": "int"
                if all(isinstance(v, int) for v in (low, high))
                else "float",
                "low": low,
                "high": high,
            }
        normalized[name] = specification
    return normalized


def suggest_parameters(trial, search_space: dict[str, Any]) -> dict[str, Any]:
    """Suggest parameters from numeric intervals, dictionaries, or callables.

    [200, 1000] samples integers within the inclusive interval. If either bound
    is a float, sample a float. [value] fixes a parameter (including strings,
    booleans, and None). [False, True] samples boolean choices, in either order.
    Equal bounds remain supported for compatibility.
    """
    search_space = normalize_search_space(search_space)
    values = {}
    for name, specification in search_space.items():
        if not isinstance(name, str) or not name:
            raise ValueError("search_space parameter names must be non-empty strings")
        if callable(specification):
            values[name] = specification(trial)
            continue
        if not isinstance(specification, dict):
            raise ValueError(
                f"Search-space specification for {name!r} must be a list, "
                "dictionary, or callable"
            )
        kind = specification.get("type", specification.get("kind"))
        if kind == "float":
            if "low" not in specification or "high" not in specification:
                raise ValueError(f"Float parameter {name!r} requires low and high")
            values[name] = trial.suggest_float(
                name,
                specification["low"],
                specification["high"],
                log=specification.get("log", False),
                step=specification.get("step"),
            )
        elif kind == "int":
            values[name] = trial.suggest_int(
                name,
                specification["low"],
                specification["high"],
                step=specification.get("step", 1),
                log=specification.get("log", False),
            )
        elif kind == "categorical":
            choices = specification.get("choices")
            if not choices:
                raise ValueError(f"Categorical parameter {name!r} requires choices")
            values[name] = trial.suggest_categorical(name, choices)
        else:
            raise ValueError(
                f"Unknown search-space type for {name!r}; use float, int, "
                "or categorical"
            )
    return values


def destination(save_as, save_path):
    if save_as is None:
        return "folder" if save_path is not None else "none"
    if save_as not in {"folder", "mlflow", "none"}:
        raise ValueError("save_as must be folder, mlflow, none, or None")
    return save_as


__all__ = ["OptimizationResult", "require_optuna", "suggest_parameters"]
