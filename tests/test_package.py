"""Tests for the top-level package metadata."""

import brainmodelkit


def test_version_is_exposed() -> None:
    """The package should expose its current version."""
    assert brainmodelkit.__version__ == "0.1.0"
