"""Tests for the top-level package metadata."""

from importlib.metadata import requires

from packaging.requirements import Requirement

import brainmodelkit


def test_version_is_exposed() -> None:
    """The package should expose its current version."""
    assert brainmodelkit.__version__ == "0.2.1"


def test_feature_dependencies_are_installed_by_default() -> None:
    """Feature extras must not be needed to install runtime dependencies."""
    requirements = [Requirement(value) for value in requires("brainmodelkit")]
    defaults = {
        requirement.name.lower(): requirement.specifier
        for requirement in requirements
        if requirement.marker is None
    }
    assert {"pyspark", "synapseml", "mlflow"} <= defaults.keys()
    for requirement in requirements:
        if requirement.marker is None:
            continue
        if requirement.marker.evaluate({"extra": "dev"}):
            continue
        assert defaults.get(requirement.name.lower()) == requirement.specifier
