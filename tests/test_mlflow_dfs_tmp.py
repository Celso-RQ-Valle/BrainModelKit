"""Optional Spark MLflow DFS staging without starting Spark or MLflow."""

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from brainmodelkit.persistence import mlflow as persistence
from brainmodelkit.training import TrainingResult, _common


@pytest.mark.parametrize("backend", ["spark", "sklearn"])
@pytest.mark.parametrize("path", [None, "/tmp/mlflow", "/Volumes/c/s/v/tmp"])
def test_logging_options_and_existing_environment(monkeypatch, backend, path):
    monkeypatch.setenv("MLFLOW_DFS_TMP", "/existing/staging")
    before = dict(os.environ)
    flavor = Mock()
    monkeypatch.setattr(persistence, "optional_import", Mock(return_value=flavor))
    persistence._log_mlflow_model("model", backend, "sig", mlflow_dfs_tmp=path)
    expected = {"artifact_path": "model", "signature": "sig"}
    if backend == "spark" and path is not None:
        expected["dfs_tmpdir"] = path
    flavor.log_model.assert_called_once_with("model", **expected)
    assert dict(os.environ) == before


@pytest.mark.parametrize("backend", ["spark", "sklearn"])
@pytest.mark.parametrize("path", [None, "hdfs://cluster/mlflow"])
def test_finalization_forwards_only_spark_option(monkeypatch, backend, path):
    tracking = Mock()
    tracking.start_run.return_value = nullcontext(
        SimpleNamespace(info=SimpleNamespace(run_id="run"))
    )
    tracking.active_run.return_value = None
    monkeypatch.setattr(_common, "optional_import", Mock(return_value=tracking))
    monkeypatch.setattr(_common, "_write_reports", Mock())
    logger = Mock()
    monkeypatch.setattr(_common, "_log_mlflow_model", logger)
    model = Mock(stages=[Mock()])
    result = TrainingResult(model, None, None, {}, [], None)
    _common.finalize_run(
        result,
        "run",
        backend,
        ["x"],
        "y",
        {},
        "unused",
        "mlflow",
        None,
        False,
        None,
        mlflow_dfs_tmp=path,
    )
    expected = {"mlflow_dfs_tmp": path} if backend == "spark" and path else {}
    logger.assert_called_once_with(model, backend, None, **expected)


@pytest.mark.parametrize(
    "backend,message,translated",
    [
        (
            "spark",
            "UC volume path must be provided to save, log or load SparkML models",
            True,
        ),
        ("spark", "permission denied", False),
        ("spark", "tracking server unavailable", False),
        ("sklearn", "UC volume path must be provided", False),
    ],
)
def test_specific_error_guidance(monkeypatch, backend, message, translated):
    error = RuntimeError(message)
    flavor = Mock()
    flavor.log_model.side_effect = error
    monkeypatch.setattr(persistence, "optional_import", Mock(return_value=flavor))
    with pytest.raises(RuntimeError) as caught:
        persistence._log_mlflow_model("model", backend)
    if translated:
        assert caught.value.__cause__ is error
        assert "mlflow_dfs_tmp=" in str(caught.value)
        assert "MLFLOW_DFS_TMP" in str(caught.value)
        assert "not normal local or Colab" in str(caught.value)
    else:
        assert caught.value is error


@pytest.mark.parametrize("path", [None, "/Volumes/c/s/v/tmp"])
def test_spark_public_parameter_reaches_finalization(monkeypatch, path):
    pytest.importorskip("pyspark")
    from brainmodelkit.training import pyspark as trainer

    frame = MagicMock(columns=["x", "y"])
    frame.filter.return_value.limit.return_value.count.return_value = 0
    frame.select.return_value.distinct.return_value.count.return_value = 2
    monkeypatch.setattr(trainer, "F", MagicMock())
    monkeypatch.setattr(trainer, "vector_to_array", MagicMock())
    monkeypatch.setattr(trainer, "VectorAssembler", Mock())
    monkeypatch.setattr(trainer, "validate_spark_persistence", Mock())
    monkeypatch.setattr(_common, "optional_import", Mock())
    fitted = Mock(stages=[Mock(featureImportances=[1.0])])
    pipeline = Mock()
    pipeline.return_value.fit.return_value = fitted
    monkeypatch.setattr(trainer, "Pipeline", pipeline)
    monkeypatch.setattr(
        trainer,
        "auc_gini",
        Mock(return_value=Mock(first=Mock(return_value={"auc": 1.0, "gini": 1.0}))),
    )
    monkeypatch.setattr(
        trainer,
        "ks_ntile",
        Mock(return_value=Mock(first=Mock(return_value={"KS": 1.0}))),
    )
    finish = Mock()
    monkeypatch.setattr(trainer, "finalize_run", finish)
    model = Mock()
    model.copy.return_value.extractParamMap.return_value = {}
    trainer.train_model(
        "run",
        "y",
        ["x"],
        frame,
        frame,
        model=model,
        save_model_to="mlflow",
        mlflow_dfs_tmp=path,
    )
    assert finish.call_args.kwargs["mlflow_dfs_tmp"] == path


@pytest.mark.parametrize("destination", ["folder", "none"])
def test_dfs_option_ignored_outside_mlflow(monkeypatch, tmp_path, destination):
    forbidden = Mock(side_effect=AssertionError("MLflow must not be called"))
    monkeypatch.setattr(_common, "optional_import", forbidden)
    monkeypatch.setattr(_common, "_log_mlflow_model", forbidden)
    saver = Mock()
    monkeypatch.setattr(_common, "_save_spark_model", saver)
    model = Mock(stages=[Mock()])
    result = TrainingResult(model, None, None, {}, [], None)
    _common.finalize_run(
        result,
        "run",
        "spark",
        ["x"],
        "y",
        {},
        tmp_path,
        destination,
        "spark",
        False,
        None,
        mlflow_dfs_tmp="hdfs://cluster/staging",
    )
    forbidden.assert_not_called()
    if destination == "folder":
        assert "dfs_tmpdir" not in saver.call_args.kwargs
        assert "mlflow_dfs_tmp" not in saver.call_args.kwargs
    else:
        saver.assert_not_called()
