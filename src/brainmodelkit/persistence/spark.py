"""Persistence using Spark's own model writer."""

from ._common import SPARK_FORMATS, optional_import, validate_format, write_metadata


def _save_spark_model(
    model, path, format="spark", *, overwrite=False, save_metadata=True, **metadata
):
    validate_format(format, SPARK_FORMATS)
    path = str(path)
    if format == "spark":
        writer = model.write()
        if overwrite:
            writer = writer.overwrite()
        writer.save(path)
    else:
        if overwrite:
            raise ValueError("overwrite=True is supported only for save_format='spark'")
        optional_import("mlflow.spark", "mlflow").save_model(model, path)
    if save_metadata:
        write_metadata(model, path, format, **metadata)
