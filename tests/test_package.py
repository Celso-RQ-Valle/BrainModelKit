"""Tests for the top-level package metadata."""

from importlib.metadata import requires

from packaging.requirements import Requirement

import brainmodelkit


def test_version_is_exposed() -> None:
    """The package should expose its current version."""
    assert brainmodelkit.__version__ == "0.2.1"


def test_runtime_dependencies_are_optional() -> None:
    """The base install must not select an execution or integration stack."""
    requirements = [Requirement(value) for value in requires("brainmodelkit")]
    assert not [
        requirement for requirement in requirements if requirement.marker is None
    ]


def test_execution_extras_are_independent() -> None:
    """Backend extras do not pull MLflow or unrelated integrations."""
    requirements = [Requirement(value) for value in requires("brainmodelkit")]
    for extra in ("pandas", "pyspark"):
        selected = {
            requirement.name.lower()
            for requirement in requirements
            if requirement.marker and requirement.marker.evaluate({"extra": extra})
        }
        assert "mlflow" not in selected
    assert "mlflow" in {
        requirement.name.lower()
        for requirement in requirements
        if requirement.marker and requirement.marker.evaluate({"extra": "mlflow"})
    }
