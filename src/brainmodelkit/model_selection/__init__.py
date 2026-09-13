"""Reusable model-selection utilities for the Pandas and Spark backends."""

from dataclasses import dataclass
from typing import Any


@dataclass
class CrossValidationResult:
    """Inspectable cross-validation output.

    ``fold_metrics`` and ``summary`` are backend-native tables for Pandas and
    small Python record lists for Spark.  Fold models are omitted by default.
    """

    fold_metrics: Any
    summary: Any
    models: list[Any] | None = None


__all__ = ["CrossValidationResult"]
