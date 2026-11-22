"""Optuna-based model optimization for supported dataframe backends."""

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


def suggest_parameters(trial, search_space: dict[str, Any]) -> dict[str, Any]:
    """Suggest parameters from candidate lists, ranges, or callable specifications.

    A list contains explicit choices: [200, 1000] samples either endpoint,
    not the integers between them. A singleton list fixes a parameter.
    """
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("search_space must be a non-empty dictionary")
    values = {}
    for name, specification in search_space.items():
        if not isinstance(name, str) or not name:
            raise ValueError("search_space parameter names must be non-empty strings")
        if callable(specification):
            values[name] = specification(trial)
            continue
        if isinstance(specification, list):
            if not specification:
                raise ValueError(f"Candidate list for {name!r} must not be empty")
            if any(
                value is not None and not isinstance(value, (str, bool, int, float))
                for value in specification
            ):
                raise ValueError(
                    f"Candidate list for {name!r} must contain scalar values"
                )
            values[name] = trial.suggest_categorical(name, specification)
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
