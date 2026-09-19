"""Persistence using Spark's own model writer."""

from ._common import SPARK_FORMATS, optional_import, validate_format, write_metadata


def validate_spark_persistence(session, destination, *, path=None):
    """Preflight the Windows local writer; MLflow chooses its own filesystem."""
    if destination != "folder":
        return
    jvm = getattr(session, "_jvm", None)
    if jvm is None:  # Spark Connect does not expose the driver's JVM.
        return
    driver_os = str(jvm.java.lang.System.getProperty("os.name"))
    if not driver_os.startswith("Windows"):
        return
    if path is not None:
        filesystem = jvm.org.apache.hadoop.fs.Path(str(path)).getFileSystem(
            session.sparkContext._jsc.hadoopConfiguration()
        )
        if str(filesystem.getUri().getScheme()) != "file":
            return
    # Hadoop's local permission operations prefer NativeIO over shell helpers.
    if jvm.org.apache.hadoop.io.nativeio.NativeIO.isAvailable():
        return

    from py4j.protocol import Py4JJavaError

    try:
        jvm.org.apache.hadoop.util.Shell.getWinUtilsPath()
    except Py4JJavaError as exc:
        raise RuntimeError(
            "Spark's Windows local writer requires Hadoop winutils when "
            "native Hadoop support is unavailable. "
            "Configure HADOOP_HOME with compatible bin/winutils.exe and "
            "native Hadoop libraries before starting Spark, then restart "
            "the Python process/notebook kernel. "
            "Use save_model_to='none' to train without "
            "saving, or run Spark in a configured Linux/WSL environment. "
            "See README: Spark on Windows."
        ) from exc


def _raise_hadoop_save_error(exc):
    """Explain a real Hadoop failure, including errors wrapped by MLflow."""
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if any(
            token in message for token in ("hadoop_home", "hadoop.home.dir", "winutils")
        ):
            raise RuntimeError(
                "Spark could not save the model because Windows Hadoop is not "
                "configured. Set HADOOP_HOME to a compatible Hadoop directory "
                "containing bin/winutils.exe, restart the notebook kernel, and "
                "retry. MLflow can also require this for Spark model staging. "
                "Use save_model_to='none' to train without persistence, "
                "or run Spark on Linux/WSL."
            ) from exc
        current = current.__cause__ or current.__context__


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
        _raise_hadoop_save_error(exc)
        raise
    if save_metadata:
        write_metadata(model, path, format, **metadata)
