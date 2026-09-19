"""Log fitted models through the appropriate MLflow flavor."""

from ._common import optional_import
from .spark import _raise_hadoop_save_error


def _log_mlflow_model(model, backend, signature=None):
    flavor = optional_import(f"mlflow.{backend}", "mlflow")
    try:
        return flavor.log_model(model, artifact_path="model", signature=signature)
    except Exception as exc:
        if backend == "spark":
            _raise_hadoop_save_error(exc)
        raise
