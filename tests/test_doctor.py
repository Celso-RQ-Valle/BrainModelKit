"""Doctor contracts using synthetic metadata and existing session objects."""

import builtins
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import brainmodelkit as bmk
from brainmodelkit.environment import capabilities as rules
from brainmodelkit.environment import detection


@pytest.fixture
def environment(monkeypatch):
    versions = {}

    def version(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(detection.metadata, "version", version)
    monkeypatch.setattr(detection, "_existing_session", lambda: None)
    monkeypatch.setattr(detection.shutil, "which", lambda name: None)
    return versions


def pandas_ready(versions):
    versions.update(
        {
            name: rules.MINIMUMS[name]
            for name in ("pandas", "numpy", "scipy", "scikit-learn")
        }
    )


def spark_session(monkeypatch, *, available=True, version="4.1.0"):
    jvm = Mock()
    jvm.java.lang.System.getProperty.return_value = "17.0.1"
    jvm.scala.util.Properties.versionNumberString.return_value = "2.13.16"
    jvm.java.lang.Class.forName.return_value.getName.return_value = "LightGBMClassifier"
    if not available:
        jvm.java.lang.Class.forName.side_effect = RuntimeError("restricted or missing")
    context = Mock()
    context._jsc.sc.return_value.isStopped.return_value = False
    session = SimpleNamespace(version=version, _jvm=jvm, _sc=context)
    monkeypatch.setattr(detection, "_existing_session", lambda: session)
    return jvm


def test_minimal_environment(environment, capsys):
    report = bmk.doctor(verbose=True)
    assert report["capabilities"]["Pandas backend"] == rules.MISSING
    assert report["capabilities"]["PySpark backend"] == rules.MISSING
    assert "BrainModelKit Doctor" in capsys.readouterr().out


def test_pandas_without_mlflow_or_lightgbm(environment):
    pandas_ready(environment)
    report = bmk.doctor(backend="pandas")
    assert report["result"] == rules.READY
    assert report["capabilities"]["MLflow persistence"] == rules.MISSING
    assert bmk.doctor()["recommended_path"] == "Pandas"
    assert bmk.doctor(backend="pandas", model="lightgbm")["result"] == rules.MISSING


@pytest.mark.parametrize(
    "version,status",
    [
        ("3.4.4", rules.INCOMPATIBLE),
        ("3.5.0", rules.READY),
        ("4.1.0", rules.READY),
        ("4.1.0+vendor.1", rules.READY),
        ("3.5.0rc1", rules.PARTIAL),
        ("unknown", rules.PARTIAL),
    ],
)
def test_pyspark_version(environment, monkeypatch, version, status):
    environment.update(pyspark=version, numpy="2.0", scipy="1.14")
    spark_session(monkeypatch, version=version)
    report = bmk.doctor(backend="pyspark")
    assert report["result"] == status
    if status == rules.READY:
        assert any("Keep the existing PySpark" in r for r in report["recommendations"])


def test_installed_spark_without_session_is_partial(environment):
    environment.update(pyspark="4.1.0", numpy="2.0", scipy="1.14")
    assert bmk.doctor(backend="pyspark")["result"] == rules.PARTIAL


@pytest.mark.parametrize("available", [True, False])
def test_spark_lightgbm_jvm(environment, monkeypatch, available):
    environment.update(
        pyspark="4.1.0", numpy="2.0", scipy="1.14", synapseml="1.1.3", mlflow="3.0.0"
    )
    jvm = spark_session(monkeypatch, available=available)
    report = bmk.doctor(backend="pyspark", model="lightgbm", save_model_to="mlflow")
    assert report["result"] == (rules.READY if available else rules.PARTIAL)
    assert report["packages"]["synapseml"] == rules.READY
    assert jvm.java.lang.Class.forName.call_args.args[1] is False
    assert any("Preserve the existing MLflow" in r for r in report["recommendations"])
    if not available:
        assert "Spark LightGBM" in report["blocking_capabilities"]


def test_mlflow_blocks_only_requested_operation(environment):
    pandas_ready(environment)
    assert bmk.doctor(backend="pandas", save_model_to="none")["result"] == rules.READY
    report = bmk.doctor(backend="pandas", save_model_to="mlflow")
    assert report["blocking_capabilities"] == {"MLflow persistence": rules.MISSING}
    assert any("brainmodelkit[mlflow]" in r for r in report["recommendations"])
    environment["mlflow"] = "3.1.0"
    assert bmk.doctor(backend="pandas", save_model_to="mlflow")["result"] == rules.READY


def test_optional_metadata_failure(environment, monkeypatch):
    monkeypatch.setattr(detection.metadata, "version", Mock(side_effect=RuntimeError))
    report = bmk.doctor(verbose=True)
    assert set(report["packages"].values()) == {rules.PARTIAL}


def test_no_imports_mutations_or_processes(environment, monkeypatch):
    # Warm standard-library imports before disabling imports and writes.
    bmk.doctor()
    before = dict(os.environ)
    forbidden = Mock(side_effect=AssertionError("unexpected side effect"))
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    importer = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"pyspark", "pandas", "mlflow", "synapse", "numpy"}:
            raise AssertionError("optional import")
        return importer(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    bmk.doctor(backend="pyspark", model="lightgbm", save_model_to="mlflow")
    forbidden.assert_not_called()
    assert dict(os.environ) == before


def test_session_lookup_never_constructs_or_calls_getactive(monkeypatch):
    module = ModuleType("pyspark.sql.session")
    session = object()
    cls = Mock()
    cls._activeSession = session
    cls._instantiatedSession = None
    module.SparkSession = cls
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert detection._existing_session() is session
    cls.assert_not_called()
    cls.getActiveSession.assert_not_called()
    cls.builder.getOrCreate.assert_not_called()


def test_restricted_jvm_is_partial(environment, monkeypatch):
    environment.update(pyspark="4.1.0", numpy="2.0", scipy="1.14", synapseml="1.1.3")

    class Restricted:
        version = "4.1.0"

        @property
        def _jvm(self):
            raise RuntimeError("access denied")

    monkeypatch.setattr(detection, "_existing_session", Restricted)
    report = bmk.doctor(backend="pyspark", model="lightgbm", verbose=True)
    assert report["result"] == rules.PARTIAL
    assert report["environment"]["jvm_error"] == "RuntimeError"


def test_runtime_version_checked_separately(environment, monkeypatch):
    environment.update(pyspark="4.1.0", numpy="2.0", scipy="1.14")
    spark_session(monkeypatch, version="3.4.0")
    report = bmk.doctor(backend="pyspark")
    assert report["packages"]["pyspark"] == rules.READY
    assert report["result"] == rules.INCOMPATIBLE


def test_packaging_rules_do_not_drift():
    from packaging.requirements import Requirement

    requirements = [Requirement(r) for r in metadata.requires("brainmodelkit")]
    for name, minimum in rules.MINIMUMS.items():
        if name != "python":
            assert any(
                r.name.lower() == name.lower() and str(r.specifier) == f">={minimum}"
                for r in requirements
            )
    assert any(
        r.name == "synapseml" and str(r.specifier) == f"=={rules.SYNAPSE_VERSION}"
        for r in requirements
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "invalid"},
        {"backend": "pandas", "model": "invalid"},
        {"model": "lightgbm"},
        {"save_model_to": "mlflow"},
        {"backend": "pandas", "save_model_to": "invalid"},
    ],
)
def test_invalid_requests(kwargs):
    with pytest.raises(ValueError):
        bmk.doctor(**kwargs)


def test_clean_interpreter_without_site_packages():
    source = str(Path(__file__).resolve().parents[1] / "src")
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import brainmodelkit as bmk; "
        "r = bmk.doctor(verbose=True); "
        "assert r['capabilities']['Pandas backend'] == 'NOT AVAILABLE'; "
        "assert 'pyspark' not in sys.modules; assert 'mlflow' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-S", "-c", script, source],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", ["stopped", "mismatch", "unknown_wrapper"])
def test_unverified_spark_prerequisites(environment, monkeypatch, mode):
    environment.update(pyspark="4.1.0", numpy="2.0", scipy="1.14", synapseml="1.1.3")
    spark_session(monkeypatch)
    session = detection._existing_session()
    if mode == "stopped":
        session._sc._jsc = None
    elif mode == "mismatch":
        session.version = "4.0.2"
    else:
        environment["synapseml"] = "9.0.0"
    report = bmk.doctor(backend="pyspark", model="lightgbm")
    assert report["result"] == rules.PARTIAL
    assert any("Keep the existing PySpark" in r for r in report["recommendations"])


def test_notebook_session_reference(monkeypatch):
    for name in ("pyspark.sql.session", "pyspark.sql.connect.session"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    session = type("SparkSession", (), {"__module__": "pyspark.sql.session"})()
    main = ModuleType("__main__")
    main.spark = session
    monkeypatch.setitem(sys.modules, "__main__", main)
    assert detection._existing_session() is session


def test_general_prefers_existing_spark(environment, monkeypatch):
    pandas_ready(environment)
    environment.update(pyspark="4.1.0", mlflow="3.0.0")
    spark_session(monkeypatch)
    report = bmk.doctor()
    assert report["recommended_path"] == "Existing Spark environment"
    assert report["result"] == rules.READY
