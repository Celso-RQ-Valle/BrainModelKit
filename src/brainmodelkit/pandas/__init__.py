"""Pandas-specific functionality for BrainModelKit."""

__all__ = [
    "ExplanationResult",
    "LogisticCalibrator",
    "ModelExplainer",
    "PredictionPipeline",
    "ScoreAligner",
    "build_autoregressive_features",
    "build_features",
    "simulate_bureau_history",
    "temporal_features",
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
        return getattr(import_module(f"brainmodelkit.{module}.pandas"), name)
    if name in __all__:
        from . import temporal

        return getattr(temporal, name)
    raise AttributeError(name)
