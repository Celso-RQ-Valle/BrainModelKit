"""Cube rollups, metric values, missing keys, and backend behavior."""

import math
import os
import sys

import pandas as pd
import pytest

from brainmodelkit.cube_analysis.pandas import calculate_metrics, calculate_ntile


def test_pandas_cube():
    df = pd.DataFrame(
        {
            "prediction": [0.9, 0.1] * 4,
            "reverse": [0.1, 0.9] * 4,
            "label": [1, 0] * 4,
            "segment": ["A"] * 4 + [None] * 4,
            "period": [1, 1, 2, 2] * 2,
        }
    )
    original = df.copy(deep=True)
    result = calculate_metrics(
        df, ["segment", "period"], ["prediction", "reverse"], "label"
    )
    assert result.columns.tolist() == [
        "segment",
        "period",
        "KS",
        "AUC",
        "Gini",
        "score",
    ]
    assert len(result) == 18
    assert result.KS.eq(1).all()
    assert result.loc[result.score.eq("prediction"), "AUC"].eq(1).all()
    assert result.loc[result.score.eq("reverse"), "Gini"].eq(-1).all()
    assert result.segment.isna().sum() == 6
    pd.testing.assert_frame_equal(df, original)


def test_empty_and_single_class():
    df = pd.DataFrame({"score": [1.0], "target": [1], "segment": ["A"]})
    assert calculate_metrics(df)[["KS", "AUC", "Gini"]].isna().all().all()
    result = calculate_metrics(df.iloc[:0], "segment")
    assert len(result) == 1
    assert result.segment.iloc[0] == "Geral"
    assert result[["KS", "AUC", "Gini"]].isna().all().all()


@pytest.mark.parametrize(
    "options, error",
    [
        ({"scores": []}, ValueError),
        ({"grupos": ["segment", "segment"]}, ValueError),
        ({"grupos": ["score"]}, ValueError),
        ({"scores": ["missing"]}, KeyError),
        ({"scores": 3}, TypeError),
    ],
)
def test_invalid_options(options, error):
    df = pd.DataFrame({"score": [1], "target": [1], "segment": ["A"]})
    with pytest.raises(error):
        calculate_metrics(df, **options)


def test_spark_cube_without_pandas(monkeypatch):
    from pyspark.sql import DataFrame, SparkSession

    from brainmodelkit.cube_analysis.pyspark import calculate_metrics as spark_cube
    from brainmodelkit.cube_analysis.pyspark import calculate_ntile as spark_ntile

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("CubeTests")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    try:
        df = spark.createDataFrame(
            [(0.9, 1, None), (0.1, 0, None), (None, 1, "missing")],
            "prediction double, label int, segment string",
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("Cube must not convert to pandas")

        monkeypatch.setattr(DataFrame, "toPandas", forbidden)
        result = spark_cube(df, "segment", ["prediction"], "label")
        assert result.columns == ["segment", "KS", "AUC", "Gini", "score"]
        rows = {row.segment: row for row in result.collect()}
        assert set(rows) == {"Geral", None, "missing"}
        for key in ["Geral", None]:
            assert (rows[key].KS, rows[key].AUC, rows[key].Gini) == (1, 1, 1)
        assert math.isnan(rows["missing"].KS)
        assert math.isnan(rows["missing"].AUC)
        empty = spark_cube(df.limit(0), "segment", "prediction", "label").collect()
        assert len(empty) == 1
        assert math.isnan(empty[0].KS)
        tiles = spark_ntile(df, "segment", "prediction", "label", n_tiles=1)
        tile_rows = {row.segment: row for row in tiles.collect()}
        assert set(tile_rows) == {"Geral", None}
        for row in tile_rows.values():
            assert row.n_tile == 1
            assert row.total_volume == 2
            assert row.total_events == row.total_non_events == 1
            assert row.event_rate == 0.5
            assert (row.minimum_range, row.maximum_range) == (0.1, 0.9)
        assert spark_ntile(df.limit(0), "segment", "prediction", "label").count() == 0
        ascending = (
            spark_ntile(
                df, scores="prediction", target="label", n_tiles=2, ascending=True
            )
            .orderBy("n_tile")
            .collect()
        )
        assert [row.total_events for row in ascending] == [0, 1]
    finally:
        spark.stop()


@pytest.mark.parametrize("ascending", [False, True])
def test_ntile_cube(ascending):
    df = pd.DataFrame(
        {
            "prediction": [1.0, 2.0, 3.0, 4.0] * 2,
            "reverse": [4.0, 3.0, 2.0, 1.0] * 2,
            "label": [0, 0, 1, 1] * 2,
            "segment": ["A"] * 4 + [None] * 4,
            "period": [1, 1, 2, 2] * 2,
        }
    )
    original = df.copy(deep=True)
    result = calculate_ntile(
        df,
        ["segment", "period"],
        ["prediction", "reverse"],
        "label",
        n_tiles=2,
        ascending=ascending,
    )
    assert len(result) == 36
    assert result.total_volume.sum() == len(df) * 4 * 2
    assert result.total_events.sum() == df.label.sum() * 4 * 2
    assert result.total_non_events.sum() == 4 * 4 * 2
    assert result.segment.isna().sum() == 12
    overall = result[
        result.segment.eq("Geral")
        & result.period.eq("Geral")
        & result.score.eq("prediction")
    ]
    assert overall.event_rate.tolist() == ([0, 1] if ascending else [1, 0])
    assert overall.total_volume.tolist() == [4, 4]
    assert calculate_ntile(df.iloc[:0], "segment", "prediction", "label").empty
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize(
    "options, error",
    [
        ({"n_tiles": 0}, ValueError),
        ({"n_tiles": True}, TypeError),
        ({"ascending": "yes"}, TypeError),
        ({"grupos": "n_tile"}, ValueError),
    ],
)
def test_ntile_validation(options, error):
    df = pd.DataFrame({"score": [1.0], "target": [1], "n_tile": [1]})
    with pytest.raises(error):
        calculate_ntile(df, **options)
