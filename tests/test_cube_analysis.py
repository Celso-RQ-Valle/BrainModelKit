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


@pytest.mark.parametrize("function", [calculate_metrics, calculate_ntile])
def test_optional_group_columns_and_named_parameters(function):
    df = pd.DataFrame({"prediction": [0.1, 0.9], "label": [0, 1]})
    options = {"df": df, "score_columns": "prediction", "target_column": "label"}
    overall = function(**options)
    for groups in [None, []]:
        pd.testing.assert_frame_equal(
            function(**options, group_columns=groups), overall
        )


@pytest.mark.parametrize(
    "options, error",
    [
        ({"score_columns": []}, ValueError),
        ({"group_columns": ["segment", "segment"]}, ValueError),
        ({"group_columns": ["score"]}, ValueError),
        ({"score_columns": ["missing"]}, KeyError),
        ({"score_columns": 3}, TypeError),
    ],
)
def test_invalid_options(options, error):
    df = pd.DataFrame({"score": [1], "target": [1], "segment": ["A"]})
    with pytest.raises(error):
        calculate_metrics(df, **options)


def test_spark_cube_without_pandas(monkeypatch):
    from pyspark.sql import SparkSession

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
            raise AssertionError("Cube must not collect rows or convert to pandas")

        frame_class = type(df)
        monkeypatch.setattr(frame_class, "toPandas", forbidden)
        # Building cubes must not collect group keys or metric values.
        # The validation action count must be independent of group count.
        original_count = frame_class.count
        counts = []

        def counted(frame):
            counts.append(1)
            return original_count(frame)

        with monkeypatch.context() as guard:
            guard.setattr(frame_class, "collect", forbidden)
            guard.setattr(frame_class, "count", counted)
            result = spark_cube(
                df=df,
                group_columns="segment",
                score_columns=["prediction"],
                target_column="label",
            )
        assert len(counts) == 1
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
        counts.clear()
        with monkeypatch.context() as guard:
            guard.setattr(frame_class, "collect", forbidden)
            guard.setattr(frame_class, "count", counted)
            tiles = spark_ntile(
                df=df,
                group_columns="segment",
                score_columns="prediction",
                target_column="label",
                n_tiles=1,
            )
        assert len(counts) == 2
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
                df,
                score_columns="prediction",
                target_column="label",
                n_tiles=2,
                ascending=True,
            )
            .orderBy("n_tile")
            .collect()
        )
        assert [row.total_events for row in ascending] == [0, 1]

        # Compare tile approximation with existing metrics, including NaN
        # scores (KS intentionally retains its existing handling of NaN).
        from brainmodelkit.metrics.pyspark import auc_gini, ks_ntile

        difficult = spark.createDataFrame(
            [
                (0.9, 1, "A"),
                (0.7, 0, "A"),
                (0.5, 1, "A"),
                (0.2, 0, "A"),
                (float("nan"), 0, "A"),
                (0.8, 1, None),
                (None, 0, "empty"),
            ],
            "score double, target int, segment string",
        )
        actual = spark_cube(difficult, "segment", n_tiles=2).collect()
        for subset in [None, "segment"]:
            expected_auc = auc_gini(df=difficult, group_by=subset, n_tiles=2).collect()
            expected_ks = ks_ntile(
                dataframe=difficult, group_columns=subset, n_tiles=2
            ).collect()
            ks_by_key = {r.segment if subset else "Geral": r.KS for r in expected_ks}
            for row in expected_auc:
                key = row.segment if subset else "Geral"
                found = next(r for r in actual if r.segment == key)
                for value, expected in [
                    (found.AUC, row.auc),
                    (found.Gini, row.gini),
                    (found.KS, ks_by_key.get(key, float("nan"))),
                ]:
                    if math.isnan(expected):
                        assert math.isnan(value)
                    else:
                        assert value == pytest.approx(round(expected, 5))

        for function in [spark_cube, spark_ntile]:
            with pytest.raises(ValueError, match="binary"):
                function(spark.createDataFrame([(0.1, 2)], "score double, target int"))
        with pytest.raises(ValueError, match="finite"):
            spark_ntile(
                spark.createDataFrame([(float("inf"), 1)], "score double, target int")
            )
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
        ({"group_columns": "n_tile"}, ValueError),
    ],
)
def test_ntile_validation(options, error):
    df = pd.DataFrame({"score": [1.0], "target": [1], "n_tile": [1]})
    with pytest.raises(error):
        calculate_ntile(df, **options)
