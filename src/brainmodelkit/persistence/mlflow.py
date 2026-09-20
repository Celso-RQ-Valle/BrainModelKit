"""Log fitted models through the appropriate MLflow flavor."""

from ._common import optional_import
from .spark import _raise_hadoop_save_error


def _log_mlflow_model(
    model, backend, signature=None, *, mlflow_dfs_tmp: str | None = None
):
    flavor = optional_import(f"mlflow.{backend}", "mlflow")
    options = {}
    if backend == "spark" and mlflow_dfs_tmp is not None:
        options["dfs_tmpdir"] = mlflow_dfs_tmp
    try:
        return flavor.log_model(
            model, artifact_path="model", signature=signature, **options
        )
    except Exception as exc:
        if backend == "spark":
            if "uc volume path must be provided" in str(exc).lower():
                raise RuntimeError(
                    "MLflow Spark persistence requires a Unity Catalog Volume in "
                    "this Databricks environment. Provide "
                    "mlflow_dfs_tmp='/Volumes/<catalog>/<schema>/<volume>/mlflow_tmp' "
                    "or configure the MLFLOW_DFS_TMP environment variable. This "
                    "requirement is specific to environments such as Databricks "
                    "Serverless/shared compute, not normal local or Colab usage."
                ) from exc
            _raise_hadoop_save_error(exc)
        raise
