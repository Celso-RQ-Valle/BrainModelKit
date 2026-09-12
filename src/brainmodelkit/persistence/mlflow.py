"""Log fitted models through the appropriate MLflow flavor."""

from ._common import optional_import


def _log_mlflow_model(model, backend, signature=None):
    flavor = optional_import(f"mlflow.{backend}", "mlflow")
    return flavor.log_model(model, artifact_path="model", signature=signature)
