"""Tests for PySpark KS metrics."""

import math
import os
import sys
from collections.abc import Iterator

import pytest
from pyspark.sql import SparkSession

from brainmodelkit.metrics.pyspark import ks_ntile


@pytest.fixture(scope="module")
def spark_session() -> Iterator[SparkSession]:
    """Create one local Spark session for metric integration tests."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    session = (
        SparkSession.builder.master("local[1]")
        .appName("BrainModelKitKSTests")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_ks_ntile_rejects_an_invalid_tile_count() -> None:
    """At least two tiles are required to compare distributions."""
    with pytest.raises(ValueError, match="at least 2"):
        ks_ntile("score", None, None, "target", n_tiles=1)  # type: ignore[arg-type]


def test_ks_ntile_reports_missing_dataframe() -> None:
    with pytest.raises(ValueError, match="dataframe is required"):
        ks_ntile()


def test_ks_ntile_calculates_overall_and_grouped_ks(
    spark_session: SparkSession,
) -> None:
    """Native Spark calculation should return expected binned KS values."""
    rows = [("complete", float(index), int(index > 50)) for index in range(1, 101)]
    rows.extend(("one_class", float(index), 1) for index in range(1, 11))
    dataframe = spark_session.createDataFrame(
        rows,
        ["segment", "score", "target"],
    )

    complete = dataframe.where("segment = 'complete'")
    assert pytest.approx(1.0) == ks_ntile(dataframe=complete).first().KS
    with pytest.raises(ValueError, match="Missing columns"):
        ks_ntile(dataframe=complete, target="missing")
    overall_ks = ks_ntile("score", complete, None, "target").first().KS
    grouped_rows = ks_ntile(
        "score",
        dataframe,
        "segment",
        "target",
    ).collect()
    grouped_ks = {row.segment: row.KS for row in grouped_rows}

    assert overall_ks == pytest.approx(1.0)
    assert grouped_ks["complete"] == pytest.approx(1.0)
    assert math.isnan(grouped_ks["one_class"])
