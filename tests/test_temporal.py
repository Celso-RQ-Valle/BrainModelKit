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
@pytest.mark.parametrize("api", ["build_features", "build_autoregressive_features"])
def test_autoregressive_base_spine_contract(backend, api, request):
    from importlib import import_module

    build = getattr(import_module(f"brainmodelkit.{backend}"), api)
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
    if api == "build_features":
        for name in list(expected.columns):
            if name not in spine.columns and name.endswith("_date"):
                expected[name] = (
                    pd.to_datetime(expected.as_of) - expected[name]
                ).dt.days
                expected.rename(columns={name: name + "_diff_days"}, inplace=True)
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


@pytest.mark.parametrize("backend", ["pandas", "pyspark"])
def test_generic_features_and_deprecation(backend, request):
    from importlib import import_module

    module = import_module(f"brainmodelkit.{backend}")
    rows = [
        ("a", "2024-02-29", -20.0, None),
        ("a", "2024-03-01", 10000.0, 25000.0),
        ("a", "2024-04-01", 99999.0, 99999.0),
    ]
    refs = [("a", "2024-03-31"), ("a", "2024-03-31"), ("missing", "2024-03-31")]
    if backend == "pyspark":
        spark = request.getfixturevalue("spark")
        base = spark.createDataFrame(
            rows,
            "account string, observation_date string, revenue double, balance double",
        )
        spine = spark.createDataFrame(refs, "account string, dt_ref string")
    else:
        base = pd.DataFrame(
            rows, columns=["account", "observation_date", "revenue", "balance"]
        )
        spine = pd.DataFrame(refs, columns=["account", "dt_ref"])
    options = dict(
        key_cols="account", value_cols=["revenue", "balance"], windows=[1], lags=[1]
    )
    result = module.build_features(base, spine, **options)
    with pytest.warns(DeprecationWarning, match="use build_features") as warnings:
        legacy = module.build_autoregressive_features(
            base, spine, date_col="observation_date", **options
        )
    assert warnings[0].filename == __file__
    if backend == "pyspark":
        actual = [r.asDict() for r in result.orderBy("account").collect()]
        old = [r.asDict() for r in legacy.orderBy("account").collect()]
        assert old[0]["revenue__current_date"] == date(2024, 3, 1)
        assert "revenue__current_date_diff_days" not in legacy.columns
    else:
        assert legacy.iloc[0]["revenue__current_date"] == pd.Timestamp("2024-03-01")
        assert "revenue__current_date_diff_days" not in legacy.columns
        actual = result.to_dict("records")
    assert len(actual) == 3
    assert actual[0]["revenue__current"] == 10000
    assert actual[0]["revenue__lag_1_months"] == -20
    assert actual[0]["revenue__sum_1_months"] == 10000
    assert actual[0]["balance__current"] == 25000
    assert actual[2]["balance__count_1_months"] == 0
    assert actual[0]["revenue__current_date_diff_days"] == 30
    assert actual[0]["revenue__lag_1_months_date_diff_days"] == 31
    assert actual[0]["balance__current_date_diff_days"] == 30
    assert pd.isna(actual[2]["revenue__current_date_diff_days"])
    assert not any(name.endswith("_date") for name in result.columns)
    if backend == "pyspark":
        from pyspark.sql import functions as F

        collision = spine.withColumn("revenue__current_date_diff_days", F.lit(0))
    else:
        collision = spine.assign(revenue__current_date_diff_days=0)
    with pytest.raises(ValueError, match="already exist"):
        module.build_features(base, collision, **options)


# Integrated optional families use the same fixture on both backends.
def relationship_inputs():
    history = pd.DataFrame(
        [
            ("a", "2026-01-15", 20.0, 100.0, 10.0),
            ("a", "2026-02-15", 60.0, 200.0, 30.0),
            ("a", "2026-02-20", 80.0, 0.0, None),
            ("a", "2026-02-22", 90.0, None, 40.0),
            ("a", "2026-02-24", None, 400.0, None),
            ("a", "2026-03-01", 999.0, 1.0, 999.0),
        ],
        columns=["id", "observation_date", "balance", "limit", "second"],
    )
    spine = pd.DataFrame(
        [
            ("a", "2026-02-28", "main"),
            ("a", "2026-02-28", "main"),
            ("a", "2026-04-30", "stale"),
            ("missing", "2026-02-28", "missing"),
        ],
        columns=["id", "dt_ref", "payload"],
    )
    return spine, history


def relationship_build(backend, request, spine, history, **options):
    from importlib import import_module

    if backend == "pyspark":
        spark = request.getfixturevalue("spark")
        spark.conf.set("spark.sql.ansi.enabled", "true")
        base = spark.createDataFrame(
            [
                tuple(None if pd.isna(v) else v for v in row)
                for row in history.itertuples(index=False, name=None)
            ],
            "id string, observation_date string, balance double, "
            "limit double, second double",
        )
        target = spark.createDataFrame(
            list(spine.itertuples(index=False, name=None)), list(spine.columns)
        )
    else:
        base, target = history, spine
    result = import_module(f"brainmodelkit.{backend}").build_features(
        base, target, key_cols="id", **options
    )
    assert len(result.columns) == len(set(result.columns))
    if backend == "pyspark":
        actual = pd.DataFrame(
            [row.asDict() for row in result.collect()], columns=result.columns
        )
        for name in actual.columns:
            if name not in spine.columns:
                actual[name] = pd.to_numeric(actual[name])
        return actual
    return result


@pytest.mark.parametrize("backend", ["pandas", "pyspark"])
@pytest.mark.parametrize(
    "relative,reference",
    [(False, None), (True, None), (False, "limit"), (True, "limit")],
)
def test_integrated_relationship_options(backend, relative, reference, request):
    from brainmodelkit.pandas import build_features

    spine, history = relationship_inputs()
    options = dict(
        value_cols=["balance", "second", "limit"],
        windows=[1, 2],
        lags=[1, 6],
        relative_historico=relative,
        ref_col=reference,
    )
    actual = relationship_build(backend, request, spine, history, **options)
    expected = build_features(history, spine, key_cols="id", **options)
    assert list(actual.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(
        actual.sort_values("payload").reset_index(drop=True),
        expected.sort_values("payload").reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        actual[spine.columns].sort_values("payload").reset_index(drop=True),
        spine.sort_values("payload").reset_index(drop=True),
    )
    row = actual[actual.payload == "main"].iloc[0]
    assert row.balance__current == 90  # No exact reference-date observation.
    assert row.balance__current_date_diff_days == 6
    assert row.balance__lag_1_months == 20
    assert row.balance__current_minus_mean_2_months == pytest.approx(27.5)
    assert not any("limit__over__limit" in name for name in actual)
    if not relative and reference is None:
        legacy = temporal_features(
            spine,
            history,
            "id",
            options["value_cols"],
            windows=[1, 2],
            lags=[1, 6],
            _date_diff=True,
        )
        pd.testing.assert_frame_equal(expected, legacy)
    if relative:
        assert row.balance__current_vs_lag_1_months_diff == 70
        assert row.balance__current_vs_lag_1_months_ratio == 4.5
        assert row.balance__current_vs_lag_1_months_pct_change == 350
        assert row.balance__range_2_months == 70
        assert row.balance__current_vs_median_2_months_diff == 20
        assert row.balance__current_vs_min_2_months_diff == 70
        assert row.balance__current_vs_max_2_months_diff == 0
        assert row.balance__current_vs_mean_2_months_ratio == pytest.approx(90 / 62.5)
        assert row.balance__standardized_deviation_2_months == pytest.approx(
            27.5 / pd.Series([20.0, 60.0, 80.0, 90.0]).std()
        )
        assert row.balance__coefficient_of_variation_2_months == pytest.approx(
            pd.Series([20.0, 60.0, 80.0, 90.0]).std() / 62.5
        )
        assert pd.isna(row.balance__current_vs_lag_6_months_ratio)
        assert "balance__current_vs_mean_2_months_diff" not in actual
    else:
        assert not any("_pct_change" in name for name in actual)
    if reference:
        prefix = "balance__over__limit__"
        assert row[prefix + "current"] == pytest.approx(0.3)
        assert row[prefix + "lag_1_months"] == pytest.approx(0.2)
        assert row[prefix + "current_date_diff_days"] == 13
        assert row[prefix + "lag_1_months_date_diff_days"] == 44
        assert row[prefix + "mean_2_months"] == pytest.approx(0.25)
        assert row[prefix + "sum_2_months"] == pytest.approx(0.5)
        assert row[prefix + "count_2_months"] == 2
        assert pd.isna(row[prefix + "std_1_months"])
        assert row["second__over__limit__current"] == pytest.approx(0.15)
        assert row.limit__current == 400
        if relative:
            assert row[prefix + "current_vs_lag_1_months_diff"] == pytest.approx(0.1)
            assert row[prefix + "current_vs_lag_1_months_diff_pp"] == pytest.approx(10)
            assert row[prefix + "current_vs_lag_1_months_pct_change"] == pytest.approx(
                50
            )
            assert row[prefix + "current_minus_mean_2_months"] == pytest.approx(0.05)
            assert row[prefix + "current_vs_mean_2_months_diff_pp"] == pytest.approx(5)
    else:
        assert not any("__over__" in name for name in actual)


@pytest.mark.parametrize("backend", ["pandas", "pyspark"])
@pytest.mark.parametrize(
    "aggs", [["mean"], ["sum", "count"], ["std"], ["min", "max"], []]
)
def test_relationship_partial_aggregations(backend, aggs, request):
    spine, history = relationship_inputs()
    result = relationship_build(
        backend,
        request,
        spine.iloc[:1],
        history,
        value_cols=["balance"],
        windows=[2],
        lags=[],
        aggregations=aggs,
        relative_historico=True,
        ref_col="limit",
    )
    row = result.iloc[0]
    assert row.balance__current_minus_mean_2_months == pytest.approx(27.5)
    for score in ("balance", "balance__over__limit"):
        for agg in ("mean", "median", "min", "max"):
            assert (f"{score}__current_vs_{agg}_2_months_ratio" in result) == (
                agg in aggs
            )
        assert (f"{score}__range_2_months" in result) == ({"min", "max"} <= set(aggs))
        assert f"{score}__standardized_deviation_2_months" not in result
        assert f"{score}__coefficient_of_variation_2_months" not in result
        for agg in ("sum", "count", "std"):
            assert not any(f"__current_vs_{agg}_" in name for name in result)
        assert not any("_lag_" in name for name in result)
    if aggs == ["sum", "count"]:
        assert row.balance__over__limit__sum_2_months == pytest.approx(0.5)
        assert row.balance__over__limit__count_2_months == 2


@pytest.mark.parametrize("backend", ["pandas", "pyspark"])
def test_relationship_staleness_empty_and_nulls(backend, request):
    spine, history = relationship_inputs()
    result = relationship_build(
        backend,
        request,
        spine,
        history,
        value_cols=["balance"],
        windows=[1],
        lags=[1],
        relative_historico=True,
        ref_col="limit",
        max_staleness_days=10,
    )
    main = result[result.payload == "main"].iloc[0]
    assert main.balance__current == 90
    assert pd.isna(
        main.balance__over__limit__current
    )  # Latest valid ratio is 13 days old.
    assert (
        main.balance__over__limit__count_1_months == 1
    )  # Staleness does not filter windows.
    for payload in ("stale", "missing"):
        row = result[result.payload == payload].iloc[0]
        assert pd.isna(row.balance__over__limit__current)
        assert pd.isna(row.balance__over__limit__current_date_diff_days)
        assert row.balance__over__limit__count_1_months == 0
        assert pd.isna(row.balance__over__limit__sum_1_months)
    empty = relationship_build(
        backend,
        request,
        spine.iloc[:1],
        history.iloc[:0],
        value_cols=["balance"],
        windows=[1],
        lags=[1],
        relative_historico=True,
        ref_col="limit",
    )
    assert empty.iloc[0].balance__over__limit__count_1_months == 0
    assert pd.isna(empty.iloc[0].balance__over__limit__current)


@pytest.mark.parametrize("backend", ["pandas", "pyspark"])
def test_relationship_zero_statistics_and_no_driver_collection(
    backend, request, monkeypatch
):
    from importlib import import_module

    spine, history = relationship_inputs()
    history = history.iloc[:2].assign(balance=0.0, limit=100.0, second=[-2.0, -4.0])
    spine = spine.iloc[:1]
    if backend == "pyspark":
        spark = request.getfixturevalue("spark")
        spark.conf.set("spark.sql.ansi.enabled", "true")
        base = spark.createDataFrame(
            list(history.itertuples(index=False, name=None)), list(history.columns)
        )
        target = spark.createDataFrame(
            list(spine.itertuples(index=False, name=None)), list(spine.columns)
        )
        frame_type = type(base)

        def forbidden(*args, **kwargs):
            raise AssertionError("No driver collection/conversion")

        with monkeypatch.context() as guard:
            guard.setattr(frame_type, "collect", forbidden)
            guard.setattr(frame_type, "toPandas", forbidden)
            result = import_module("brainmodelkit.pyspark").build_features(
                base,
                target,
                key_cols="id",
                value_cols=["balance", "second"],
                windows=[2],
                lags=[1],
                ref_col="limit",
                relative_historico=True,
            )
        row = result.first().asDict()
    else:
        result = import_module("brainmodelkit.pandas").build_features(
            history,
            spine,
            key_cols="id",
            value_cols=["balance", "second"],
            windows=[2],
            lags=[1],
            ref_col="limit",
            relative_historico=True,
        )
        row = result.iloc[0]
    assert pd.isna(row["balance__current_vs_lag_1_months_ratio"])
    assert pd.isna(row["balance__current_vs_mean_2_months_pct_change"])
    assert pd.isna(row["balance__standardized_deviation_2_months"])
    assert pd.isna(row["balance__coefficient_of_variation_2_months"])
    assert row["second__coefficient_of_variation_2_months"] == pytest.approx(2**0.5 / 3)


@pytest.mark.parametrize("backend", ["pandas", "pyspark"])
def test_relationship_validation(backend, request):
    from importlib import import_module

    spine, history = relationship_inputs()
    if backend == "pyspark":
        spark = request.getfixturevalue("spark")
        base = spark.createDataFrame(
            list(history.iloc[:2].itertuples(index=False, name=None)),
            list(history.columns),
        )
        target = spark.createDataFrame(
            list(spine.itertuples(index=False, name=None)), list(spine.columns)
        )
        from pyspark.sql import functions as F

        def add(frame, name, value):
            return frame.withColumn(name, F.lit(value))
    else:
        base, target = history, spine

        def add(frame, name, value):
            return frame.assign(**{name: value})

    build = import_module(f"brainmodelkit.{backend}").build_features
    options = dict(
        key_cols="id",
        value_cols=["balance"],
        windows=[2],
        lags=[1],
        relative_historico=True,
    )
    for ref in ("missing", "payload", "id", "", 3):
        with pytest.raises(ValueError, match="ref_col"):
            build(base, target, **options, ref_col=ref)
    with pytest.raises(ValueError, match="ref_col must be numeric"):
        build(add(base, "limit", "text"), target, **options, ref_col="limit")
    for name in (
        "balance__current_vs_lag_1_months_diff",
        "balance__over__limit__current",
    ):
        with pytest.raises(ValueError, match="already exist"):
            build(base, add(target, name, 0.0), **options, ref_col="limit")
    with pytest.raises(ValueError, match="collide"):
        build(
            add(base, "balance__over__limit", 0.0), target, **options, ref_col="limit"
        )
    with pytest.raises(ValueError, match="bool"):
        build(base, target, **(options | {"relative_historico": "yes"}))
    duplicate = (
        pd.concat([history, history.iloc[:1]])
        if backend == "pandas"
        else base.union(base)
    )
    with pytest.raises(ValueError, match="Duplicate"):
        build(duplicate, target, **options, ref_col="limit")


@pytest.mark.parametrize("backend", ["pandas", "pyspark"])
@pytest.mark.parametrize("unit,lag,window", [("days", 30, 60), ("weeks", 5, 9)])
def test_relationship_units_and_ratio_nulls(backend, unit, lag, window, request):
    spine, history = relationship_inputs()
    history.loc[2, "limit"] = float("nan")
    result = relationship_build(
        backend,
        request,
        spine.iloc[:1],
        history,
        value_cols=["balance"],
        windows=[window],
        lags=[lag],
        period_unit=unit,
        aggregations=["mean", "std"],
        relative_historico=True,
        ref_col="limit",
    )
    row = result.iloc[0]
    prefix = "balance__over__limit__"
    assert row[prefix + f"current_vs_lag_{lag}_{unit}_diff_pp"] == pytest.approx(10)
    assert row[prefix + f"mean_{window}_{unit}"] == pytest.approx(0.25)
    assert row[prefix + f"standardized_deviation_{window}_{unit}"] == pytest.approx(
        2**-0.5
    )
    assert row[prefix + f"coefficient_of_variation_{window}_{unit}"] == pytest.approx(
        (0.005**0.5) / 0.25
    )


def test_relationship_pandas_index_inputs_and_empty_spine():
    from brainmodelkit.pandas import build_features

    spine, history = relationship_inputs()
    spine.index = [9, 9, 2, 4]
    original_spine, original_history = spine.copy(deep=True), history.copy(deep=True)
    options = dict(
        key_cols="id",
        value_cols=["balance", "limit"],
        windows=[2],
        lags=[1],
        relative_historico=True,
        ref_col="limit",
    )
    result = build_features(history, spine, **options)
    pd.testing.assert_frame_equal(result[spine.columns], spine)
    pd.testing.assert_frame_equal(spine, original_spine)
    pd.testing.assert_frame_equal(history, original_history)
    empty = build_features(history, spine.iloc[:0], **options)
    assert empty.empty
    assert list(empty.columns) == list(result.columns)
    no_windows = build_features(
        history, spine, **(options | {"windows": [], "lags": []})
    )
    assert (
        len(no_windows.columns) == len(spine.columns) + 6
    )  # Three series, current + age.
