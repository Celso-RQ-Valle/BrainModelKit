"""Persistence using Spark's own model writer."""

from ._common import SPARK_FORMATS, optional_import, validate_format, write_metadata


def validate_spark_persistence(session, destination):
    """Detect missing Windows Hadoop helpers before fitting or opening a run."""
    if destination == "none":
        return
    jvm = getattr(session, "_jvm", None)
    if jvm is None:  # Spark Connect does not expose the driver's JVM.
        return
    driver_os = str(jvm.java.lang.System.getProperty("os.name"))
    if not driver_os.startswith("Windows"):
        return

    from py4j.protocol import Py4JJavaError

    try:
        jvm.org.apache.hadoop.util.Shell.getWinUtilsPath()
    except Py4JJavaError as exc:
        raise RuntimeError(
            "Spark model persistence on Windows requires Hadoop winutils. "
            "Configure HADOOP_HOME with compatible bin/winutils.exe and "
            "native Hadoop libraries before starting Spark, then restart "
            "the Python process/notebook kernel. MLflow also uses Spark's "
            "native writer. Use save_model_to='none' to train without "
            "saving, or run Spark in a configured Linux/WSL environment. "
            "See README: Spark on Windows."
        ) from exc


def _save_spark_model(
    model, path, format="spark", *, overwrite=False, save_metadata=True, **metadata
):
    validate_format(format, SPARK_FORMATS)
    path = str(path)
    try:
        if format == "spark":
            writer = model.write()
            if overwrite:
                writer = writer.overwrite()
            writer.save(path)
        else:
            if overwrite:
                raise ValueError(
                    "overwrite=True is supported only for save_format='spark'"
                )
            optional_import("mlflow.spark", "mlflow").save_model(model, path)
    except Exception as exc:
        message = str(exc)
        if "HADOOP_HOME" in message or "winutils" in message:
            raise RuntimeError(
                "Spark could not save the model because Windows Hadoop is not "
                "configured. Set HADOOP_HOME to a compatible Hadoop directory "
                "containing bin/winutils.exe, restart the notebook kernel, and "
                "retry. Use save_model_to='none' to train without persistence, "
                "or run Spark on Linux/WSL."
            ) from exc
        raise
    if save_metadata:
        write_metadata(model, path, format, **metadata)
