"""Load explicitly selected model formats without importing optional backends."""

import pickle

from ._common import PYTHON_FORMATS, optional_import, validate_format
from .python import _load_native_model

__all__ = ["load_model"]


def load_model(path, format="pickle", *, model_class=None, trusted=None):
    """Load a model using its declared format, never its filename extension.

    Only load trusted artifacts with pickle, joblib, cloudpickle or MLflow.
    For skops, explicitly pass any reviewed additional types via ``trusted``.
    Native and Spark formats require ``model_class``. LightGBM native files
    load as Booster objects. ONNX returns an onnxruntime InferenceSession.
    MLflow selects the Spark or sklearn flavor from its own model manifest.
    """
    validate_format(format, (*PYTHON_FORMATS, "spark"))
    path = str(path)
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
    for flavor in ("spark", "sklearn"):
        if flavor in manifest.flavors:
            return optional_import(f"mlflow.{flavor}", "mlflow").load_model(path)
    raise ValueError("MLflow model must contain a Spark or sklearn flavor")
