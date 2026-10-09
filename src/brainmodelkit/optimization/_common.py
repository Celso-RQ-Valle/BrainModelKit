"""Shared metric naming, constrained sampling, and final trial selection."""

import math
import warnings


def metric_values(train, cv, oot):
    """Signed shifts retain the historical OOT minus reference convention."""
    if not train or set(train) != set(cv) or set(cv) != set(oot):
        raise ValueError(
            "Missing metrics: train, CV, and OOT must expose the same metrics"
        )
    values = {}
    for metric in cv:
        for prefix, source in (("train", train), ("cv", cv), ("oot", oot)):
            try:
                value = float(source[metric])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Missing or non-finite metric: {prefix}_{metric}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"Missing or non-finite metric: {prefix}_{metric}")
            values[f"{prefix}_{metric}"] = value
        values[metric] = values[f"cv_{metric}"]
        values[f"{metric}_shift"] = values[f"oot_{metric}"] - values[f"cv_{metric}"]
        for reference in ("train", "cv"):
            shift = values[f"oot_{metric}"] - values[f"{reference}_{metric}"]
            values[f"{reference}_oot_{metric}_shift"] = shift
            values[f"{reference}_oot_{metric}_abs_shift"] = abs(shift)
    return values


def validate_options(
    metrics,
    objective_metric,
    direction,
    n_trials,
    target_metric,
    target_limit,
    target_operator,
    search_space,
):
    requested = tuple(str(m).lower() for m in metrics)
    if not requested or any(m not in {"ks", "auc", "gini"} for m in requested):
        raise ValueError("metrics must contain only ks, auc, and gini")
    available = metric_values(*[{m: 0.0 for m in requested}] * 3)
    if not isinstance(objective_metric, str):
        raise ValueError("objective_metric must be a metric name")
    objective_metric = objective_metric.lower()
    if objective_metric not in available:
        raise ValueError("objective_metric must name a metric included in metrics")
    supplied = (
        target_metric is not None,
        target_limit is not None,
        target_operator is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "Supply target_metric, target_limit, and target_operator together"
        )
    if target_metric is not None:
        if not isinstance(target_metric, str):
            raise ValueError("target_metric must be a metric name")
        target_metric = target_metric.lower()
        if target_metric not in available:
            raise ValueError("target_metric must name a metric included in metrics")
        if target_operator not in {"<=", ">="}:
            raise ValueError("target_operator must be <= or >=")
        if (
            isinstance(target_limit, bool)
            or not isinstance(target_limit, (int, float))
            or not math.isfinite(target_limit)
        ):
            raise ValueError("target_limit must be a finite number in decimal units")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be maximize or minimize")
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise ValueError("n_trials must be an integer >= 1")
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("search_space must be a non-empty dictionary")
    if "lgbm_backend" in search_space:
        raise ValueError("lgbm_backend is a backend selector, not a search parameter")
    return requested, objective_metric, target_metric


def record_trial(trial, values, params, target_metric, limit, operator):
    for name, value in values.items():
        trial.set_user_attr(name, value)
    residual = 0.0
    if target_metric is not None:
        residual = values[target_metric] - limit
        if operator == ">=":
            residual = -residual
    trial.set_user_attr("constraint_residual", residual)
    trial.set_user_attr("constraint_violation", max(0.0, residual))
    trial.set_user_attr("feasible", residual <= 0)
    trial.set_user_attr("parameters", params)


def constraint_values(trial):
    return (trial.user_attrs.get("constraint_residual", float("inf")),)


def make_study(optuna, name, direction, seed, constrained):
    options = {"seed": seed}
    if constrained:
        options["constraints_func"] = constraint_values
    return optuna.create_study(
        study_name=name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(**options),
    )


def select_trial(study, direction, constrained):
    completed = [
        t
        for t in study.trials
        if t.state.name == "COMPLETE" and t.value is not None and math.isfinite(t.value)
    ]
    if not completed:
        raise ValueError(
            "Optimization produced no completed trials with finite metrics"
        )

    def rank(trial):
        violation = trial.user_attrs.get("constraint_violation", 0.0)
        objective = trial.value if direction == "minimize" else -trial.value
        return (violation > 0, violation, objective, trial.number)

    selected = min(completed, key=rank)
    feasible = selected.user_attrs["feasible"]
    if constrained and not feasible:
        warnings.warn(
            f"No feasible trial: target was not achieved. Selecting least-violating "
            f"trial {selected.number} (violation="
            f"{selected.user_attrs['constraint_violation']:.6g}).",
            RuntimeWarning,
            stacklevel=2,
        )
    study.set_user_attr("selected_trial_number", selected.number)
    study.set_user_attr("constraint_satisfied", feasible if constrained else None)
    return selected


def trial_records(study):
    return [
        dict(
            trial=t.number,
            params=t.user_attrs.get("parameters", t.params),
            value=t.value,
            status=t.state.name,
            **t.user_attrs,
        )
        for t in study.trials
    ]
