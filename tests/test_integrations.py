"""Tests for optional integration namespaces."""


def test_pandas_namespace_is_importable() -> None:
    """The Pandas namespace should not require Pandas at import time."""
    from brainmodelkit import pandas

    assert pandas.__name__ == "brainmodelkit.pandas"


def test_pyspark_namespace_is_importable() -> None:
    """The PySpark namespace should not require PySpark at import time."""
    from brainmodelkit import pyspark

    assert pyspark.__name__ == "brainmodelkit.pyspark"
