"""Explicit, optional local feature attribution without model retraining."""

from brainmodelkit._probability import CapabilityError

__all__ = ["CapabilityError", "ExplanationResult"]


def __getattr__(name):
    if name == "ExplanationResult":
        from ._result import ExplanationResult

        return ExplanationResult
    raise AttributeError(name)
