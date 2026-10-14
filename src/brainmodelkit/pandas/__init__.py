"""Pandas-specific functionality for BrainModelKit."""

__all__ = [
    "build_autoregressive_features",
    "simulate_bureau_history",
    "temporal_features",
]


def __getattr__(name):
    if name in __all__:
        from . import temporal

        return getattr(temporal, name)
    raise AttributeError(name)
