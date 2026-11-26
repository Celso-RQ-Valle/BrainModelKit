"""PySpark-specific functionality for BrainModelKit."""

from .credit import simulate_credit_data
from .temporal import (
    build_autoregressive_features,
    build_features,
    simulate_bureau_history,
    temporal_features,
)
from .validation import validate_external_dataframe

__all__ = [
    "ExplanationResult",
    "LogisticCalibrator",
    "ModelExplainer",
    "PredictionPipeline",
    "ScoreAligner",
    "build_autoregressive_features",
    "build_features",
    "simulate_bureau_history",
    "simulate_credit_data",
    "temporal_features",
    "validate_external_dataframe",
]


def __getattr__(name):
    if name in {
        "LogisticCalibrator",
        "ScoreAligner",
        "ModelExplainer",
        "PredictionPipeline",
        "ExplanationResult",
    }:
        from importlib import import_module

        module = (
            "calibration"
            if name == "LogisticCalibrator"
            else "scoring"
            if name == "ScoreAligner"
            else "prediction"
            if name == "PredictionPipeline"
            else "explainability"
        )
        return getattr(import_module(f"brainmodelkit.{module}.pyspark"), name)
    raise AttributeError(name)
