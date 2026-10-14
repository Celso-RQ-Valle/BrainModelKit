"""Point-in-time contracts shared by pandas and Spark."""

import os
import sys
from datetime import date

import pandas as pd
import pytest

from brainmodelkit.pandas import simulate_bureau_history, temporal_features


def inputs():
    spine = pd.DataFrame(
        [
            ("001", "a", "2026-10-13", "first"),
            ("001", "a", "2026-10-13", "first"),
            ("001", "a", "2026-03-31", "early"),
            ("001", "b", "2026-10-13", "other"),
        ],
        columns=["id", "branch", "dt_ref", "payload"],
    )
    history = pd.DataFrame(
        [
            ("001", "a", "2026-02-13", 10.0, None),
            ("001", "a", "2026-02-28", 20.0, 2.0),
            ("001", "a", "2026-08-13", 30.0, 3.0),
            ("001", "a", "2026-09-14", 40.0, 4.0),
            ("001", "a", "2026-10-05", 50.0, None),
            ("001", "a", "2026-10-14", 999.0, 999.0),
        ],
        columns=["id", "branch", "observation_date", "score", "second"],
    )
    return spine, history


def kwargs():
    return dict(
        entity_cols=["id", "branch"],
        score_cols=["score", "second"],
        windows=[1, 8],
        lags=[1, 2, 8],
    )


def test_boundaries_nulls_and_cardinality():
    spine, history = inputs()
    spine.index = [7, 7, 9, 2]
    result = temporal_features(spine, history, **kwargs())
    pd.testing.assert_frame_equal(result[spine.columns], spine)
    row = result.iloc[0]
    assert row.score__current == 50
    assert row.score__current_date == pd.Timestamp("2026-10-05")
    assert row.score__lag_1_months == 30
    assert row.score__lag_2_months == 30
    assert row.score__lag_8_months == 10
    assert row.score__count_8_months == 4
    assert row.score__mean_8_months == 35
    assert row.score__current_minus_mean_8_months == 15
    assert row.second__current == 4
    assert result.iloc[2].score__count_1_months == 0  # Feb 28 excluded
    assert pd.isna(result.iloc[2].score__sum_1_months)
    assert result.iloc[3].score__count_8_months == 0
    assert pd.isna(result.iloc[3].score__current)
    assert pd.isna(result.iloc[0].second__std_1_months)


@pytest.mark.parametrize("unit,k", [("days", 8), ("weeks", 1)])
def test_units_and_staleness(unit, k):
    spine, history = inputs()
    result = temporal_features(
        spine,
        history,
        ["id", "branch"],
        ["score"],
        unit=unit,
        windows=[k],
        lags=[k],
        max_staleness_days=7,
    )
    assert pd.isna(result.iloc[0].score__current)
    assert result.iloc[0][f"score__lag_{k}_{unit}"] == 50


def test_duplicates_and_validation():
    spine, history = inputs()
    with pytest.raises(ValueError, match="Duplicate"):
        temporal_features(spine, pd.concat([history, history.iloc[:1]]), **kwargs())
    for override in (
        {"unit": "years"},
        {"lags": [-1]},
        {"windows": [1, 1]},
        {"aggregations": ["bad"]},
        {"max_staleness_days": -1},
    ):
        with pytest.raises(ValueError):
            temporal_features(spine, history, **(kwargs() | override))
    empty = temporal_features(spine.iloc[:0], history, **kwargs())
    assert empty.empty
    result = temporal_features(spine, history.iloc[:0], **kwargs())
    assert result.score__current.isna().all()


def test_simulation_determinism_cohorts_and_identifiers():
    customers = pd.DataFrame({"id": [f"{i:05}" for i in range(500)], "branch": "a"})
    args = dict(
        entity_cols=["id", "branch"],
        initial_date="2025-01-01",
        final_date="2025-12-31",
        seed=17,
    )
    result = simulate_bureau_history(customers, **args)
    other = simulate_bureau_history(customers.iloc[::-1], **args)
    order = ["id", "observation_date"]
    pd.testing.assert_frame_equal(
        result.sort_values(order).reset_index(drop=True),
        other.sort_values(order).reset_index(drop=True),
    )
    assert result.bureau_score.between(0, 1000).all()
    assert result.id.str.len().eq(5).all()
    counts = result.groupby("id").size()
    assert counts.eq(12).mean() > 0.5
    assert counts.lt(12).any()
    starts = result.groupby("id").observation_date.min()
    assert starts.gt(pd.Timestamp("2025-01-31")).any()
    # A customer present from January with fewer than 12 rows has missing periods,
    # rather than simply being a late entrant.
    assert (starts.eq(pd.Timestamp("2025-01-31")) & counts.lt(12)).any()
    different = simulate_bureau_history(customers, **(args | {"seed": 18}))
    assert not result.equals(different)
    complete = simulate_bureau_history(
        customers.iloc[:2], **args, late_entry_fraction=0, missing_customer_fraction=0
    )
    assert len(complete) == 24
    assert complete.observation_date.dt.is_month_end.all()


@pytest.mark.parametrize(
    "frequency,count", [("daily", 31), ("weekly", 5), ("monthly", 1)]
)
def test_alignment(frequency, count):
    result = simulate_bureau_history(
        pd.DataFrame({"id": ["001"]}),
        "id",
        "2025-01-01",
        "2025-01-31",
        frequency,
        late_entry_fraction=0,
        missing_customer_fraction=0,
    )
    assert len(result) == count


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    session = (
        SparkSession.builder.master("local[1]")
        .appName("TemporalTests")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_spark_parity(spark, monkeypatch):
    from pyspark.sql import DataFrame

    from brainmodelkit.pyspark import temporal_features as spark_features

    def forbidden(*args, **kwargs):
        raise AssertionError("No driver conversion")

    monkeypatch.setattr(DataFrame, "toPandas", forbidden)
    spine, history = inputs()
    hrows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in history.itertuples(index=False, name=None)
    ]
    sh = spark.createDataFrame(
        hrows,
        "id string, branch string, observation_date string, "
        "score double, second double",
    )
    ss = spark.createDataFrame(
        list(spine.itertuples(index=False, name=None)), list(spine.columns)
    )
    expected = temporal_features(spine, history, **kwargs())
    actual = spark_features(ss, sh, **kwargs()).collect()
    assert len(actual) == len(expected)
    for row in actual:
        target = expected[expected.payload == row.payload].iloc[0]
        for name in expected.columns:
            value = target[name]
            if pd.isna(value):
                assert row[name] is None
            elif isinstance(value, pd.Timestamp):
                assert row[name] == value.date()
            elif isinstance(value, (float, int)):
                assert row[name] == pytest.approx(value)
            else:
                assert row[name] == value
    with pytest.raises(ValueError, match="Duplicate"):
        spark_features(ss, sh.union(sh), **kwargs())


def test_spark_simulation_parity(spark):
    from brainmodelkit.pyspark import simulate_bureau_history as simulate

    customers = pd.DataFrame({"id": ["001", "002"], "branch": [1, 2]})
    args = dict(
        entity_cols=["id", "branch"],
        initial_date=date(2025, 1, 1),
        final_date=date(2025, 12, 31),
        seed=9,
    )
    expected = simulate_bureau_history(customers, **args)
    sc = spark.createDataFrame([("001", 1), ("002", 2)], ["id", "branch"])
    actual = (
        simulate(sc.repartition(2), **args).orderBy("id", "observation_date").collect()
    )
    assert len(actual) == len(expected)
    for row, target in zip(actual, expected.itertuples(index=False), strict=True):
        assert row.id == target.id
        assert row.observation_date == target.observation_date.date()
        assert row.bureau_score == pytest.approx(target.bureau_score)


@pytest.mark.parametrize(
    "override",
    [
        {"frequency": "yearly"},
        {"seed": 1.5},
        {"noise_std": -1},
        {"score_min": 1000},
        {"persistence": 1.1},
        {"late_entry_fraction": 0.4, "missing_customer_fraction": 0.2},
        {"missing_period_probability": -1},
        {"final_date": "2024-01-01"},
    ],
)
def test_invalid_simulation(override):
    args = dict(initial_date="2025-01-01", final_date="2025-12-31") | override
    with pytest.raises(ValueError):
        simulate_bureau_history(pd.DataFrame({"id": ["001"]}), "id", **args)


def test_leap_year_and_zero_noise():
    history = pd.DataFrame(
        {
            "id": ["001", "001"],
            "observation_date": ["2024-02-29", "2024-03-01"],
            "score": [5.0, 7.0],
        }
    )
    spine = pd.DataFrame({"id": ["001"], "dt_ref": ["2024-03-31"]})
    result = temporal_features(
        spine, history, "id", "score", windows=[1], lags=[1], aggregations=["count"]
    )
    assert result.iloc[0].score__lag_1_months == 5
    assert result.iloc[0].score__count_1_months == 1
    assert result.iloc[0].score__current_minus_mean_1_months == 0
    simulation = simulate_bureau_history(
        spine,
        "id",
        "2024-01-01",
        "2024-03-31",
        noise_std=0,
        late_entry_fraction=0,
        missing_customer_fraction=0,
    )
    assert simulation.bureau_score.nunique() == 1
    assert simulation.observation_date.iloc[1] == pd.Timestamp("2024-02-29")
    empty = simulate_bureau_history(spine, "id", "2024-01-01", "2024-01-02")
    assert empty.empty


def test_spark_empty_stale_and_validation(spark, monkeypatch):
    try:
        from pyspark.sql.classic.dataframe import DataFrame
    except ImportError:
        from pyspark.sql import DataFrame

    from brainmodelkit.pyspark import temporal_features as spark_features

    spine = spark.createDataFrame([("001", "2024-03-31")], "id string, dt_ref string")
    hist = spark.createDataFrame(
        [("001", "2024-02-29", 5.0), ("001", "2024-03-01", 7.0)],
        "id string, observation_date string, score double",
    )
    original_collect = DataFrame.collect

    def forbidden(*args, **kwargs):
        raise AssertionError("No driver collection/conversion during feature building")

    with monkeypatch.context() as guard:
        guard.setattr(DataFrame, "collect", forbidden)
        guard.setattr(DataFrame, "toPandas", forbidden)
        result = spark_features(
            spine,
            hist,
            "id",
            "score",
            windows=[1],
            lags=[1],
            max_staleness_days=0,
            aggregations=["count"],
        )
    row = original_collect(result)[0]
    assert row.score__current is None
    assert row.score__lag_1_months == 5
    assert row.score__count_1_months == 1
    for unit in ("days", "weeks"):
        row = spark_features(
            spine,
            hist,
            "id",
            "score",
            unit=unit,
            windows=[1],
            lags=[],
            max_staleness_days=1,
        ).first()
        assert row.score__current is None
        assert row[f"score__count_1_{unit}"] == 0
        assert row[f"score__sum_1_{unit}"] is None
    assert spark_features(spine.limit(0), hist, "id", "score").count() == 0
    row = spark_features(spine, hist.limit(0), "id", "score").first()
    assert row.score__current is None
    assert row.score__count_3_months == 0
    invalid = spark.createDataFrame([("001", "not-a-date")], "id string, dt_ref string")
    with pytest.raises(ValueError, match="dates"):
        spark_features(invalid, hist, "id", "score")


@pytest.mark.parametrize("backend", ["pandas", "pyspark"])
def test_autoregressive_base_spine_contract(backend, request):
    from importlib import import_module

    build = import_module(f"brainmodelkit.{backend}").build_autoregressive_features
    spine, history = inputs()
    # Conflicting spine scores must be retained but never used as feature values.
    spine = spine.assign(score=-999.0).rename(columns={"dt_ref": "as_of"})
    history = history.rename(columns={"observation_date": "event_date"})
    options = dict(
        key_cols=["id", "branch"],
        date_col="event_date",
        reference_date_col="as_of",
        value_cols=["score", "second"],
        period_unit="weeks",
        windows=[1, 8],
        lags=[1, 2],
        aggregations=["mean", "sum", "count"],
        max_staleness_days=10,
    )
    expected = temporal_features(
        spine,
        history,
        ["id", "branch"],
        ["score", "second"],
        observation_col="event_date",
        reference_col="as_of",
        unit="weeks",
        windows=[1, 8],
        lags=[1, 2],
        aggregations=["mean", "sum", "count"],
        max_staleness_days=10,
    )
    if backend == "pyspark":
        spark = request.getfixturevalue("spark")
        base = spark.createDataFrame(
            [
                tuple(None if pd.isna(v) else v for v in row)
                for row in history.itertuples(index=False, name=None)
            ],
            "id string, branch string, event_date string, score double, second double",
        )
        target = spark.createDataFrame(
            list(spine.itertuples(index=False, name=None)), list(spine.columns)
        )
        actual = pd.DataFrame(
            [r.asDict() for r in build(base, target, **options).collect()]
        )
        for col in actual.columns:
            if col.endswith("_date"):
                actual[col] = pd.to_datetime(actual[col])
            elif col not in spine.columns:
                actual[col] = pd.to_numeric(actual[col])
    else:
        original_base, original_spine = history.copy(deep=True), spine.copy(deep=True)
        actual = build(df_base=history, df_spine=spine, **options)
        pd.testing.assert_frame_equal(history, original_base)
        pd.testing.assert_frame_equal(spine, original_spine)
    pd.testing.assert_frame_equal(
        actual.sort_values("payload").reset_index(drop=True),
        expected.sort_values("payload").reset_index(drop=True),
    )
    assert len(actual) == len(spine)
    assert actual.score.eq(-999).all()
    first = actual[actual.payload == "first"].iloc[0]
    assert first.score__current == 50
    assert first.score__lag_1_weeks == 50
    assert pd.isna(first.score__lag_2_weeks)
    assert first.score__mean_8_weeks == 45
    assert pd.isna(actual[actual.payload == "early"].iloc[0].score__current)


@pytest.mark.parametrize("backend", ["pandas", "pyspark"])
def test_autoregressive_default_columns(backend, request):
    from importlib import import_module

    build = import_module(f"brainmodelkit.{backend}").build_autoregressive_features
    base_rows = [("001", "2026-10-05", 50.0), ("001", "2026-10-14", 999.0)]
    spine_rows = [("001", "2026-10-13")]
    if backend == "pyspark":
        spark = request.getfixturevalue("spark")
        base = spark.createDataFrame(
            base_rows, "cnpj string, bureau_date string, score double"
        )
        spine = spark.createDataFrame(spine_rows, "cnpj string, dt_ref string")
        row = build(df_base=base, df_spine=spine).first()
    else:
        base = pd.DataFrame(base_rows, columns=["cnpj", "bureau_date", "score"])
        spine = pd.DataFrame(spine_rows, columns=["cnpj", "dt_ref"])
        row = build(base, spine).iloc[0]
    assert row["score__current"] == 50
    assert row["score__count_3_months"] == 1
