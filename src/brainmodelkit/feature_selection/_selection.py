"""Dependency-free result and validation helpers for composable selectors.

Usage, method algorithms, parameters, and examples:
https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection.md
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from brainmodelkit.training import TrainingResult


class TrainingOutputs:
    """Read-only access to evaluation outputs, matching training and RFE."""

    training_result: TrainingResult | None

    @property
    def metrics(self) -> dict[str, float]:
        return self.training_result.metrics if self.training_result is not None else {}

    @property
    def oot_predictions(self) -> Any:
        return (
            self.training_result.oot_predictions
            if self.training_result is not None
            else None
        )

    @property
    def scoring_predictions(self) -> Any:
        return (
            self.training_result.scoring_predictions
            if self.training_result is not None
            else None
        )

    @property
    def feature_importance(self) -> list[dict]:
        return (
            self.training_result.feature_importance
            if self.training_result is not None
            else []
        )

    @property
    def run_id(self) -> str | None:
        return self.training_result.run_id if self.training_result is not None else None

    @property
    def model_uri(self) -> str | None:
        return (
            self.training_result.model_uri if self.training_result is not None else None
        )

    @property
    def save_model_to(self) -> str:
        return (
            self.training_result.save_model_to
            if self.training_result is not None
            else "none"
        )

    @property
    def model_format(self) -> str | None:
        return (
            self.training_result.model_format
            if self.training_result is not None
            else None
        )


@dataclass
class SelectionResult(TrainingOutputs):
    """Ordered feature diagnostics; large Spark diagnostics remain distributed.

    ``feature_table`` is a list of dictionaries, one per input feature.
    Selection lists preserve input order. Method-specific tables are optional.
    Existing RFE continues to return its original RFEResult.
    """

    feature_table: list[dict[str, Any]]
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)
    grouped_table: Any = None
    correlation_pairs: list[dict[str, Any]] = field(default_factory=list)
    bin_details: Any = None
    cv_results: Any = None
    history: list[dict[str, Any]] = field(default_factory=list)
    model: Any = None
    selection_model: Any = None
    training_result: TrainingResult | None = None

    @property
    def output_dir(self) -> Any:
        return (
            self.training_result.output_dir
            if self.training_result is not None
            else None
        )

    @property
    def selected_features(self) -> list[str]:
        return [r["feature"] for r in self.feature_table if r["selected"]]

    @property
    def rejected_features(self) -> list[str]:
        return [r["feature"] for r in self.feature_table if not r["selected"]]

    @property
    def ranking(self) -> dict[str, int]:
        return {
            r["feature"]: r["ranking"] for r in self.feature_table if "ranking" in r
        }

    @property
    def optimal_feature_count(self) -> int:
        return len(self.selected_features)


def resolve_frame(
    primary: Any, alias_value: Any, name: str, alias: str, *, required: bool = True
) -> Any:
    """Resolve frame aliases by identity, never DataFrame equality/truthiness."""
    if primary is not None and alias_value is not None and primary is not alias_value:
        raise ValueError(f"{name} conflicts with {alias}; pass only one spelling")
    result = primary if primary is not None else alias_value
    if required and result is None:
        raise ValueError(f"{name} is required")
    return result


def columns(
    df: Any, feature_cols: Sequence[str], target_col: str | None = None
) -> list[str]:
    if isinstance(feature_cols, str) or feature_cols is None:
        raise ValueError("feature_cols must be a non-empty sequence")
    try:
        features = list(feature_cols)
    except TypeError as exc:
        raise ValueError("feature_cols must be a non-empty sequence") from exc
    if not features:
        raise ValueError("feature_cols must be a non-empty sequence")
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("DataFrame column names must be unique")
    if any(not isinstance(f, str) for f in features) or len(set(features)) != len(
        features
    ):
        raise ValueError("feature_cols must contain unique string names")
    if target_col in features:
        raise ValueError("target_col must not be a feature")
    missing = set(features + ([target_col] if target_col is not None else [])) - set(
        df.columns
    )
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return features


def groups(df: Any, group_by: str | Sequence[str] | None) -> list[str]:
    if group_by is None:
        return []
    names = [group_by] if isinstance(group_by, str) else list(group_by)
    columns(df, names)
    if set(names) & {"feature", "completeness", "missing_rate", "selected"}:
        raise ValueError("group_by names conflict with diagnostic output columns")
    return names


def number(
    value: float, name: str, minimum: float = 0, maximum: float | None = None
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(
            f"{name} must be >= {minimum}"
            + (f" and <= {maximum}" if maximum is not None else "")
        )


def integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def ranked(
    rows: list[dict],
    score: str,
    method: str,
    threshold: float | None = None,
    top_k: int | None = None,
    *,
    absolute: bool = False,
) -> SelectionResult:
    """Descending scores; stable input-order ties; undefined scores rejected."""
    if threshold is not None:
        number(threshold, "threshold", minimum=-float("inf"))
    if top_k is not None:
        integer(top_k, "top_k")
        if top_k > len(rows):
            raise ValueError("top_k cannot exceed the feature count")

    def value(row: dict) -> float:
        v = float(row[score])
        return abs(v) if absolute else v

    order = sorted(
        range(len(rows)),
        key=lambda i: (
            -value(rows[i]) if not math.isnan(value(rows[i])) else float("inf")
        ),
    )
    for rank, i in enumerate(order, 1):
        v = value(rows[i])
        rows[i]["ranking"] = rank
        rows[i]["selected"] = bool(
            not math.isnan(v)
            and (threshold is None or v >= threshold)
            and (top_k is None or rank <= top_k)
        )
    return SelectionResult(rows, method, {"threshold": threshold, "top_k": top_k})


def correlation_result(
    features: list[str], matrix: Any, threshold: float, method: str
) -> SelectionResult:
    kept: list[int] = []
    pairs = []
    rows = []
    for j, feature in enumerate(features):
        rejected_by = None
        for i in range(j):
            value = float(matrix[i][j])
            if math.isfinite(value) and abs(value) >= threshold:
                pairs.append(
                    {
                        "feature_1": features[i],
                        "feature_2": feature,
                        "correlation": value,
                    }
                )
                if i in kept and rejected_by is None:
                    rejected_by = features[i]
        if rejected_by is None:
            kept.append(j)
        rows.append(
            {
                "feature": feature,
                "selected": rejected_by is None,
                "rejected_by": rejected_by,
            }
        )
    return SelectionResult(
        rows,
        "correlation",
        {"threshold": threshold, "method": method},
        correlation_pairs=pairs,
    )
