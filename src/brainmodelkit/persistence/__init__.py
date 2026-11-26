"""Load explicitly selected model formats without importing optional backends."""

import pickle

from ._common import PYTHON_FORMATS, local_path, optional_import, validate_format
from .bundle import load_workflow, save_workflow
from .python import _load_native_model
from .resolution import ArtifactSpec, resolve_model


def load_model(path, format="pickle", *, model_class=None, trusted=None, flavor=None):
    """Load a model using its declared format, never its filename extension.

    Only load trusted artifacts with pickle, joblib, cloudpickle or MLflow.
    For skops, explicitly pass any reviewed additional types via ``trusted``.
    Native and Spark formats require ``model_class``. LightGBM native files
    load as Booster objects. ONNX returns an onnxruntime InferenceSession.
    MLflow prefers an installed native flavor. Select pyfunc explicitly.
    """
    validate_format(format, (*PYTHON_FORMATS, "spark"))
    path = str(path)
    if path.startswith(("runs:/", "models:/")) and format != "mlflow":
        raise ValueError("MLflow URIs require format='mlflow'")
    if local_path(path) is None and format not in {"spark", "mlflow"}:
        raise ValueError("Use an accessible local or mounted path for this format")
    if flavor is not None and format != "mlflow":
        raise ValueError("flavor applies only to MLflow")
    if format == "pickle":
        with open(path, "rb") as stream:
            return pickle.load(stream)
    if format == "cloudpickle":
        module = optional_import("cloudpickle", "persistence")
        with open(path, "rb") as stream:
            return module.load(stream)
    if format == "joblib":
        return optional_import("joblib", "persistence").load(path)
    if format == "skops":
        return optional_import("skops.io", "secure").load(path, trusted=trusted)
    if format == "native":
        return _load_native_model(path, model_class)
    if format == "spark":
        if model_class is None:
            raise ValueError("Spark loading requires model_class, e.g. PipelineModel")
        return model_class.load(path)
    if format == "onnx":
        return optional_import("onnxruntime", "onnx").InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )
    manifest = optional_import("mlflow.models", "mlflow").Model.load(path)
    native = ("spark", "sklearn", "lightgbm", "xgboost", "catboost", "onnx")
    selected = flavor or next((f for f in native if f in manifest.flavors), None)
    if selected == "pyfunc" and "python_function" in manifest.flavors:
        return optional_import("mlflow.pyfunc", "mlflow").load_model(path)
    if selected not in native or selected not in manifest.flavors:
        raise ValueError(
            f"Select a native flavor or explicit pyfunc: {list(manifest.flavors)}"
        )
    model = optional_import(f"mlflow.{selected}", "mlflow").load_model(path)
    if selected == "onnx":
        return optional_import("onnxruntime", "onnx").InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
    return model


__all__ = [
    "ArtifactSpec",
    "load_model",
    "load_workflow",
    "resolve_model",
    "save_workflow",
]
