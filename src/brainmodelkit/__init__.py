"""Tools for model development, evaluation, and analysis."""

__version__ = "0.2.1"


def doctor(
    *,
    backend=None,
    model=None,
    save_model_to=None,
    verbose=False,
    lgbm_backend="default",
):
    """Print and return read-only environment and workflow diagnostics."""
    from .environment.doctor import doctor as diagnose

    return diagnose(
        backend=backend,
        model=model,
        save_model_to=save_model_to,
        verbose=verbose,
        lgbm_backend=lgbm_backend,
    )


__all__ = ["__version__", "doctor"]
