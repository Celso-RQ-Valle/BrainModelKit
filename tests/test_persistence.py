"""Persistence contracts without requiring distributed or heavy runtimes."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from brainmodelkit.persistence import _common, load_model
from brainmodelkit.persistence.python import _save_python_model
from brainmodelkit.persistence.spark import _save_spark_model


@pytest.mark.parametrize("format", ["pickle", "joblib", "cloudpickle"])
def test_python_round_trip(tmp_path, format):
    if format != "pickle":
        pytest.importorskip(format)
    path = tmp_path / "nested" / "model"
    _save_python_model(
        {"weights": [1, 2]},
        path,
        format,
        target_col="y",
        feature_cols=["x"],
        parameters={"depth": 2, "callback": lambda: None},
    )
    assert load_model(path, format) == {"weights": [1, 2]}
    metadata = json.loads((path.parent / "model.metadata.json").read_text())
    assert metadata["save_format"] == format
    assert metadata["target_col"] == "y"
    assert metadata["feature_cols"] == ["x"]
    assert metadata["parameters"] == {"depth": 2}
    assert metadata["python_version"]
    assert metadata["brainmodelkit_version"]
    assert metadata["created_at"]


def test_no_metadata(tmp_path):
    path = tmp_path / "model"
    _save_python_model({}, path, save_metadata=False)
    assert not (tmp_path / "model.metadata.json").exists()


@pytest.mark.parametrize("save", [_save_python_model, _save_spark_model])
def test_invalid_format(tmp_path, save):
    with pytest.raises(ValueError, match="Available:"):
        save({}, tmp_path / "model", "invalid")
    with pytest.raises(ValueError, match="Available:"):
        load_model(tmp_path / "model", "invalid")


@pytest.mark.parametrize(
    "format,extra",
    [
        ("joblib", "persistence"),
        ("cloudpickle", "persistence"),
        ("skops", "secure"),
        ("onnx", "onnx"),
        ("mlflow", "mlflow"),
    ],
)
def test_missing_optional_dependency(tmp_path, monkeypatch, format, extra):
    monkeypatch.setattr(
        _common.importlib, "import_module", Mock(side_effect=ImportError("missing"))
    )
    with pytest.raises(ImportError, match=rf"brainmodelkit\[{extra}\]"):
        _save_python_model({}, tmp_path / "model", format)
    with pytest.raises(ImportError, match=rf"brainmodelkit\[{extra}\]"):
        load_model(tmp_path / "model", format)


@pytest.mark.parametrize("overwrite", [False, True])
def test_spark_writer_and_metadata(tmp_path, overwrite):
    model = Mock()
    writer = model.write.return_value
    writer.overwrite.return_value = writer
    path = tmp_path / "model"
    _save_spark_model(model, path, overwrite=overwrite, feature_cols=["x"])
    writer.save.assert_called_once_with(str(path))
    assert writer.overwrite.call_count == int(overwrite)
    metadata = json.loads((tmp_path / "model.metadata.json").read_text())
    assert metadata["save_format"] == "spark"
    assert metadata["feature_cols"] == ["x"]
    cls = Mock()
    assert load_model(path, "spark", model_class=cls) is cls.load.return_value
    cls.load.assert_called_once_with(str(path))


def test_spark_remote_path():
    model = Mock()
    _save_spark_model(model, "hdfs://cluster/models/model")
    model.write.return_value.save.assert_called_once_with("hdfs://cluster/models/model")


@pytest.mark.parametrize("format", ["native", "spark"])
def test_class_required(format):
    with pytest.raises(ValueError, match="requires model_class"):
        load_model("model", format)


@pytest.mark.parametrize("framework", ["lightgbm", "xgboost", "catboost"])
def test_native_round_trip(tmp_path, framework):
    class Booster:
        def __init__(self, model_file=None):
            self.path = model_file

        def save_model(self, path):
            self.path = path

        def load_model(self, path):
            self.path = path

    Booster.__module__ = framework
    model = Booster()
    path = tmp_path / "native"
    _save_python_model(model, path, "native", save_metadata=False)
    assert load_model(path, "native", model_class=Booster).path == str(path)
    assert model.path == str(path)


def test_lightgbm_wrapper(tmp_path):
    wrapper = type("LGBMClassifier", (), {"__module__": "lightgbm.sklearn"})()
    wrapper.booster_ = Mock()
    _save_python_model(wrapper, tmp_path / "model", "native", save_metadata=False)
    wrapper.booster_.save_model.assert_called_once_with(str(tmp_path / "model"))


def test_lightgbm_native_predictions(tmp_path):
    lgb = pytest.importorskip("lightgbm")
    import numpy as np

    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    model = lgb.LGBMClassifier(min_child_samples=1, verbosity=-1).fit(
        features, [0, 0, 1, 1]
    )
    path = tmp_path / "model.txt"
    _save_python_model(model, path, "native")
    loaded = load_model(path, "native", model_class=lgb.Booster)
    np.testing.assert_allclose(
        loaded.predict(features), model.booster_.predict(features)
    )


def test_onnx_and_skops_contracts(tmp_path, monkeypatch):
    backend = Mock()
    monkeypatch.setattr(_common.importlib, "import_module", Mock(return_value=backend))
    backend.to_onnx.return_value.SerializeToString.return_value = b"onnx"
    path = tmp_path / "model"
    _save_python_model("model", path, "onnx", input_example="sample")
    backend.to_onnx.assert_called_once_with("model", "sample")
    assert path.read_bytes() == b"onnx"
    assert load_model(path, "onnx") is backend.InferenceSession.return_value
    _save_python_model("model", path, "skops")
    assert (
        load_model(path, "skops", trusted=["reviewed.Type"])
        is backend.load.return_value
    )
    backend.load.assert_called_once_with(str(path), trusted=["reviewed.Type"])


@pytest.mark.parametrize("flavor", ["spark", "sklearn"])
def test_mlflow_flavor(tmp_path, monkeypatch, flavor):
    backend = Mock()
    backend.Model.load.return_value = SimpleNamespace(flavors={flavor: {}})
    importer = Mock(return_value=backend)
    monkeypatch.setattr(_common.importlib, "import_module", importer)
    save = _save_spark_model if flavor == "spark" else _save_python_model
    path = tmp_path / "model"
    save("model", path, "mlflow")
    backend.save_model.assert_called_once_with("model", str(path))
    assert load_model(path, "mlflow") is backend.load_model.return_value
    assert importer.call_args.args == (f"mlflow.{flavor}",)
