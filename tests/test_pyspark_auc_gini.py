"""Spark integration tests for tile-based ROC AUC and Gini."""

import math
import os
import sys
from collections.abc import Iterator
from itertools import pairwise

import pytest
from pyspark.sql import SparkSession

from brainmodelkit.metrics.pyspark import auc_gini, curve_roc, risk_table


@pytest.mark.parametrize("ascending", [False, True])
def test_risk_table_matches_pandas(
    spark_session: SparkSession, ascending: bool
) -> None:
    import pandas as pd

    from brainmodelkit.metrics.pandas import risk_table as pandas_risk_table

    rows = [(float(i), int(i > 5)) for i in range(1, 11)]
    rows += [(None, 1), (float("nan"), 0), (20.0, None)]
    df = spark_session.createDataFrame(rows, "prediction double, label int")
    result = risk_table(
        "prediction", df, "label", n_tiles=3, ascending=ascending
    ).toPandas()
    expected = pandas_risk_table(
        "prediction",
        pd.DataFrame(rows, columns=["prediction", "label"]),
        "label",
        n_tiles=3,
        ascending=ascending,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_risk_table_empty_ties_and_single_class(spark_session: SparkSession) -> None:
    df = spark_session.createDataFrame([(5.0, 1)] * 3, "score double, target int")
    result = risk_table(df=df).collect()
    assert [r.n_tile for r in result] == [1, 2, 3]
    assert all(r.minimum_range == r.maximum_range == 5 for r in result)
    assert all(
        r.event_rate == 1 and r.total_volume == 1 and r.total_non_events == 0
        for r in result
    )
    assert risk_table(df=df, n_tiles=1).first().total_volume == 3
    empty = risk_table(df=df.limit(0))
    assert empty.columns == list(result[0].asDict())
    assert empty.count() == 0


def test_risk_table_spark_validation(spark_session: SparkSession) -> None:
    df = spark_session.createDataFrame([(1.0, 2)], "score double, target int")
    with pytest.raises(ValueError, match="binary"):
        risk_table(df=df)
    with pytest.raises(TypeError):
        risk_table()
    with pytest.raises(TypeError):
        risk_table(df=df, n_tiles=True)
    with pytest.raises(ValueError):
        risk_table(df=df, n_tiles=0)
    with pytest.raises(KeyError):
        risk_table(df=df, target_column="missing")
    invalid = spark_session.createDataFrame(
        [(float("inf"), 1)], "score double, target int"
    )
    with pytest.raises(ValueError, match="finite"):
        risk_table(df=invalid)


def test_curve_roc_matches_tile_auc(spark_session: SparkSession) -> None:
    df = spark_session.createDataFrame(
        [(0.9, 1), (0.2, 1), (0.8, 0), (0.1, 0)], "score double, target int"
    )
    points = curve_roc(df=df, n_tiles=4).collect()
    assert [(p.tile, p.fpr, p.tpr) for p in points] == [
        (0, 0, 0),
        (1, 0, 0.5),
        (2, 0.5, 0.5),
        (3, 0.5, 1),
        (4, 1, 1),
    ]
    area = sum((b.fpr - a.fpr) * (b.tpr + a.tpr) / 2 for a, b in pairwise(points))
    assert area == pytest.approx(auc_gini(df=df, n_tiles=4).first().auc)


def test_curve_roc_empty_and_missing_pairs(spark_session: SparkSession) -> None:
    df = spark_session.createDataFrame(
        [(0.9, 1.0), (0.1, 0.0), (None, 1.0), (0.5, float("nan"))],
        "prediction double, label double",
    )
    assert curve_roc("prediction", df, "label").count() == 3
    for data in [df.limit(0), df.filter("label = 1")]:
        result = curve_roc("prediction", data, "label")
        assert result.columns == ["tile", "fpr", "tpr"]
        assert result.count() == 0
    with pytest.raises(TypeError):
        curve_roc()
    with pytest.raises(KeyError):
        curve_roc(df=df)
    with pytest.raises(TypeError):
        curve_roc(df=df, n_tiles=True)
    with pytest.raises(ValueError):
        curve_roc(df=df, n_tiles=1)


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
