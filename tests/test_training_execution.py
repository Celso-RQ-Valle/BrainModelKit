"""Destination orchestration and compatibility, without heavy services."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from brainmodelkit.persistence import load_model
from brainmodelkit.training import TrainingResult
from brainmodelkit.training._common import finalize_run
from brainmodelkit.training.pandas import train_model


@pytest.fixture
def data():
    return pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0], "y": [0, 0, 1, 1]})


@pytest.mark.parametrize("backend", ["pandas", "pyspark"])
@pytest.mark.parametrize("destination", ["local", "mlflow", "none"])
def test_runs_as_routes_to_existing_execution(backend, destination, monkeypatch):
    import importlib

    module = importlib.import_module(f"brainmodelkit.training.{backend}")
    monkeypatch.setattr("brainmodelkit.training._common.optional_import", Mock())

    def stop_after_validation(*args):
        raise RuntimeError("validated")

    resolver = Mock(wraps=module.resolve_execution)
    monkeypatch.setattr(module, "resolve_execution", resolver)
    monkeypatch.setattr(module, "validate_columns", stop_after_validation)
    with pytest.raises(RuntimeError, match="validated"):
        module.train_model("test", "y", ["x"], None, None, runs_as=destination)
    assert resolver.call_args.kwargs["runs_as"] == destination
    assert (
        module.resolve_execution._mock_wraps(
            "spark" if backend == "pyspark" else "sklearn",
            "local",
            None,
            None,
            None,
            None,
            False,
            runs_as=destination,
        )[0]
        == destination
    )


@pytest.mark.parametrize("destination", ["local", "none"])
def test_runs_as_pandas_persistence(data, tmp_path, destination):
    result = train_model(
        "alias", "y", ["x"], data, data, runs_as=destination, output_dir=tmp_path
    )
    assert result.run_as == destination
    if destination == "local":
        assert Path(result.model_uri).is_file()
        assert list(load_model(result.model_uri).predict(data[["x"]])) == list(data.y)
    else:
        assert result.model_uri is None
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "options",
    [
        {"runs_as": "invalid"},
        {"runs_as": "local", "run_as": "none"},
        {"runs_as": "local", "mlflow_logging": True},
    ],
)
def test_runs_as_invalid_or_conflicting(options):
    with pytest.raises(ValueError):
        train_model("bad", "y", ["x"], None, None, **options)


def test_none_has_no_filesystem_or_serialization(data, tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("none must not perform filesystem operations or serialization")

    monkeypatch.setattr(Path, "mkdir", unexpected)
    monkeypatch.setattr("brainmodelkit.training._common._save_python_model", unexpected)
    result = train_model("none", "y", ["x"], data, data, run_as="none", output_dir=None)
    assert result.model is not None
    assert result.metrics["oot_auc"] == 1
    assert result.run_as == "none"
    assert result.output_dir is result.model_uri is result.run_id is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("backend", ["pandas", "pyspark"])
@pytest.mark.parametrize(
    "options,match",
    [
        ({"run_as": "invalid"}, "Available:"),
        ({"model_format": "invalid"}, "Available:"),
        ({"model_format": "mlflow"}, "Available:"),
        ({"signature": True}, "requires run_as='mlflow'"),
        ({"run_as": "none", "signature": True}, "requires run_as='mlflow'"),
        ({"run_as": "mlflow", "model_format": "onnx"}, "require|Available:"),
    ],
)
def test_validation_before_fit(backend, options, match):
    if backend == "pyspark":
        pytest.importorskip("pyspark")
        from brainmodelkit.training.pyspark import train_model as trainer
    else:
        trainer = train_model
    with pytest.raises(ValueError, match=match):
        trainer("bad", "y", ["x"], None, None, **options)


def test_optional_local_format(data, tmp_path):
    pytest.importorskip("joblib")
    result = train_model(
        "joblib", "y", ["x"], data, data, output_dir=tmp_path, model_format="joblib"
    )
    assert Path(result.model_uri).name == "model.joblib"
    assert list(load_model(result.model_uri, "joblib").predict(data[["x"]])) == list(
        data.y
    )


def test_default_local_needs_no_optional_persistence(data, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "brainmodelkit.persistence._common.importlib.import_module",
        Mock(side_effect=AssertionError("optional dependency import")),
    )
    result = train_model("local", "y", ["x"], data, data, output_dir=tmp_path)
    assert Path(result.model_uri).exists()


@pytest.mark.parametrize("overwrite", [False, True])
def test_spark_local_writer(tmp_path, overwrite):
    model = Mock()
    model.stages = [Mock()]
    writer = model.write.return_value
    writer.overwrite.return_value = writer
    result = TrainingResult(model, None, None, {"oot_auc": 1.0}, [], None)
    result = finalize_run(
        result,
        "spark",
        "spark",
        ["x"],
        "y",
        {"bad": float("nan"), "depth": 2},
        tmp_path,
        "local",
        "spark",
        False,
        None,
        overwrite=overwrite,
    )
    writer.save.assert_called_once_with(result.model_uri)
    assert writer.overwrite.call_count == int(overwrite)
    report = json.loads((result.output_dir / "model_info.json").read_text())
    assert report["parameters"] == {"depth": 2}
    assert report["run_id"] == result.run_id
    assert (result.output_dir / "feature_importance.csv").read_text().strip() == (
        "feature,importance"
    )


def test_local_unique_safe_names(data, tmp_path):
    results = [
        train_model("../../credit/v1", "y", ["x"], data, data, output_dir=tmp_path)
        for _ in range(2)
    ]
    assert results[0].output_dir != results[1].output_dir
    assert all(r.output_dir.parent == tmp_path for r in results)


def test_missing_mlflow_before_fit(monkeypatch):
    monkeypatch.setattr(
        "brainmodelkit.persistence._common.importlib.import_module",
        Mock(side_effect=ImportError("missing")),
    )
    with pytest.raises(ImportError, match=r"brainmodelkit\[mlflow\]"):
        train_model("bad", "y", ["x"], None, None, run_as="mlflow")


@pytest.mark.parametrize("backend", ["sklearn", "spark"])
def test_mlflow_complete_run(backend, tmp_path, monkeypatch):
    mlflow = Mock()
    mlflow.active_run.return_value = None
    mlflow.start_run.return_value = nullcontext(
        SimpleNamespace(info=SimpleNamespace(run_id="abc"))
    )
    uploaded = {}

    def capture(folder, artifact_path):
        assert artifact_path == "training_report"
        for path in Path(folder).iterdir():
            uploaded[path.name] = path.read_text()

    mlflow.log_artifacts.side_effect = capture
    monkeypatch.setattr(
        "brainmodelkit.persistence._common.importlib.import_module",
        Mock(return_value=mlflow),
    )
    model = Mock()
    model.stages = [Mock()]
    result = TrainingResult(
        model,
        None,
        None,
        {"oot_auc": 1.0, "oot_ks": float("nan")},
        [{"feature": "x", "importance": 2.0}],
        None,
    )
    result = finalize_run(
        result,
        "credit",
        backend,
        ["x"],
        "y",
        {"depth": 2},
        tmp_path,
        "mlflow",
        None,
        False,
        None,
    )
    assert result.run_id == "abc"
    assert result.run_as == "mlflow"
    assert result.model_uri == "runs:/abc/model"
    assert result.output_dir is None
    assert not list(tmp_path.iterdir())
    mlflow.log_params.assert_called_once_with({"depth": "2"})
    mlflow.log_metrics.assert_called_once_with({"oot_auc": 1.0})
    mlflow.log_model.assert_called_once_with(
        model, artifact_path="model", signature=None
    )
    tags = mlflow.set_tags.call_args.args[0]
    assert tags["target_col"] == "y"
    assert json.loads(tags["feature_cols"]) == ["x"]
    assert tags["brainmodelkit_version"]
    assert tags["python_version"]
    assert set(uploaded) == {
        "model_info.json",
        "metrics.json",
        "feature_importance.csv",
        "metadata.json",
    }
    assert json.loads(uploaded["model_info.json"])["run_id"] == "abc"
    assert json.loads(uploaded["metrics.json"])["oot_ks"] is None


def test_legacy_external_path(data, tmp_path):
    path = tmp_path / "legacy.joblib"
    pytest.importorskip("joblib")
    with pytest.warns(DeprecationWarning):
        result = train_model(
            "legacy",
            "y",
            ["x"],
            data,
            data,
            output_dir=tmp_path,
            save_path=path,
            save_format="joblib",
        )
    assert result.model_uri == str(path)
    assert (tmp_path / "legacy.joblib.metadata.json").exists()
    assert not (result.output_dir / "model.joblib").exists()
    assert list(load_model(path, "joblib").predict(data[["x"]])) == list(data.y)


def test_legacy_logging_alias(data, tmp_path, monkeypatch):
    monkeypatch.setattr("brainmodelkit.training._common.optional_import", Mock())
    finish = Mock()
    monkeypatch.setattr("brainmodelkit.training.pandas.finalize_run", finish)
    with pytest.warns(DeprecationWarning):
        train_model(
            "legacy", "y", ["x"], data, data, output_dir=tmp_path, mlflow_logging=True
        )
    assert finish.call_args.args[7] == "mlflow"


def test_legacy_conflicts(data):
    for options in (
        {"run_as": "none", "mlflow_logging": True},
        {"model_format": "pickle", "save_format": "joblib"},
        {"run_as": "none", "save_path": "model"},
        {"save_format": "mlflow"},
    ):
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError):
            train_model("bad", "y", ["x"], data, data, **options)


def test_spark_none_without_runtime(monkeypatch):
    model = Mock()
    result = TrainingResult(model, None, None, {}, [], None)
    monkeypatch.setattr(Path, "mkdir", Mock(side_effect=AssertionError("filesystem")))
    result = finalize_run(
        result, "none", "spark", ["x"], "y", {}, None, "none", None, False, None
    )
    assert result.model is model
    assert result.run_as == "none"
    assert result.output_dir is result.model_uri is result.run_id is None
    model.write.assert_not_called()
