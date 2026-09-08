"""Spark integration tests for tile-based ROC AUC and Gini."""

import math
import os
import sys
from collections.abc import Iterator

import pytest
from pyspark.sql import SparkSession

from brainmodelkit.metrics.pyspark import auc_gini


@pytest.fixture(scope="module")
def spark_session() -> Iterator[SparkSession]:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    session = (
        SparkSession.builder.master("local[1]")
        .appName("BrainModelKitAUCTests")
        .config("spark.sql.ansi.enabled", "true")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.mark.parametrize(
    ("scores", "tiles", "expected"),
    [
        ([0.9, 0.8, 0.2, 0.1], 10, 1.0),
        ([0.1, 0.2, 0.8, 0.9], 10, 0.0),
        ([0.9, 0.2, 0.8, 0.1], 2, 0.5),
        ([0.9, 0.2, 0.8, 0.1], 4, 0.75),
    ],
)
def test_overall_auc(
    spark_session: SparkSession, scores: list[float], tiles: int, expected: float
) -> None:
    df = spark_session.createDataFrame(
        list(zip(scores, [1, 1, 0, 0], strict=True)), "score double, target int"
    )
    result = auc_gini(df=df, n_tiles=tiles).first()
    assert result.asDict() == pytest.approx({"auc": expected, "gini": 2 * expected - 1})


def test_missing_pairs_and_custom_columns(spark_session: SparkSession) -> None:
    df = spark_session.createDataFrame(
        [
            (0.9, 1.0),
            (0.1, 0.0),
            (None, 1.0),
            (float("nan"), 0.0),
            (0.8, None),
            (0.7, float("nan")),
        ],
        "prediction double, label double",
    )
    result = auc_gini("prediction", df, target_column="label").first()
    assert result.asDict() == {"auc": 1.0, "gini": 1.0}


@pytest.mark.parametrize("rows", [[], [(0.9, 1.0)], [(None, None)]])
def test_undefined_auc(spark_session: SparkSession, rows: list) -> None:
    df = spark_session.createDataFrame(rows, "score double, target double")
    result = auc_gini(df=df).first()
    assert math.isnan(result.auc)
    assert math.isnan(result.gini)


@pytest.mark.parametrize("group_by", ["segment", ["segment", "region"]])
def test_grouped_auc(spark_session: SparkSession, group_by: str | list[str]) -> None:
    df = spark_session.createDataFrame(
        [
            ("A", 1, 0.9, 1),
            ("A", 1, 0.1, 0),
            (None, 1, 0.1, 1),
            (None, 1, 0.9, 0),
            ("single", 1, 0.9, 1),
            ("missing", 1, None, 0),
        ],
        "segment string, region int, score double, target int",
    )
    result = auc_gini(df=df, group_by=group_by)
    rows = {row.segment: row for row in result.collect()}
    assert rows["A"].auc == 1.0
    assert rows[None].gini == -1.0
    assert math.isnan(rows["single"].auc)
    assert math.isnan(rows["missing"].auc)
    assert result.schema["segment"].dataType == df.schema["segment"].dataType


def test_empty_grouped_schema(spark_session: SparkSession) -> None:
    df = spark_session.createDataFrame([], "segment string, score double, target int")
    result = auc_gini(df=df, group_by="segment")
    assert result.columns == ["segment", "auc", "gini"]
    assert result.count() == 0


def test_input_validation(spark_session: SparkSession) -> None:
    with pytest.raises(TypeError, match="PySpark DataFrame"):
        auc_gini()
    df = spark_session.createDataFrame([(0.9, 1)], "score double, target int")
    for tiles in [True, 2.5, "10"]:
        with pytest.raises(TypeError, match="integer"):
            auc_gini(df=df, n_tiles=tiles)
    with pytest.raises(ValueError, match="greater than or equal"):
        auc_gini(df=df, n_tiles=1)
    for group in [[], 1, [1], ("score",), ["score", "score"], "auc"]:
        with pytest.raises(ValueError):
            auc_gini(df=df, group_by=group)
    with pytest.raises(KeyError, match="missing"):
        auc_gini(df=df, target_column="missing")
    invalid = spark_session.createDataFrame([(0.9, 2)], "score double, target int")
    with pytest.raises(ValueError, match="binary"):
        auc_gini(df=invalid)
