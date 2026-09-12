"""Optional persistence for Python estimators."""

import pickle
from pathlib import Path

from ._common import PYTHON_FORMATS, optional_import, validate_format, write_metadata


def _save_native_model(model, path):
    framework = type(model).__module__.split(".")[0]
    if framework == "lightgbm":
        getattr(model, "booster_", model).save_model(path)
    elif framework in {"xgboost", "catboost"}:
        model.save_model(path)
    else:
        raise ValueError("Native persistence supports LightGBM, XGBoost and CatBoost")


def _load_native_model(path, model_class):
    if model_class is None:
        raise ValueError(
            "native loading requires model_class (use lightgbm.Booster for LightGBM)"
        )
    framework = model_class.__module__.split(".")[0]
    if framework == "lightgbm":
        if model_class.__name__ != "Booster":
            raise ValueError(
                "LightGBM native loading requires model_class=lightgbm.Booster"
            )
        return model_class(model_file=path)
    if framework in {"xgboost", "catboost"}:
        model = model_class()
        model.load_model(path)
        return model
    raise ValueError("Native persistence supports LightGBM, XGBoost and CatBoost")


def _save_python_model(
    model, path, format="pickle", *, save_metadata=True, input_example=None, **metadata
):
    validate_format(format, PYTHON_FORMATS)
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if format == "pickle":
        with open(path, "wb") as stream:
            pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    elif format == "cloudpickle":
        module = optional_import("cloudpickle", "persistence")
        with open(path, "wb") as stream:
            module.dump(model, stream)
    elif format in {"joblib", "skops"}:
        module = optional_import(
            "skops.io" if format == "skops" else "joblib",
            "secure" if format == "skops" else "persistence",
        )
        module.dump(model, path)
    elif format == "native":
        _save_native_model(model, path)
    elif format == "onnx":
        converter = optional_import("skl2onnx", "onnx")
        if input_example is None:
            raise ValueError("ONNX conversion requires input_example")
        converted = converter.to_onnx(model, input_example)
        Path(path).write_bytes(converted.SerializeToString())
    elif format == "mlflow":
        optional_import("mlflow.sklearn", "mlflow").save_model(model, path)
    if save_metadata:
        write_metadata(model, path, format, **metadata)
