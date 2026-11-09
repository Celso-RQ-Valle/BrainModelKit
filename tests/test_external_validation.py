"""External-provider matching, frozen score bands and fixed-bin density."""

import os
import sys
from datetime import date, datetime

import pytest

from brainmodelkit.pyspark import validate_external_dataframe


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    session = (
        SparkSession.builder.master("local[1]")
        .appName("ExternalValidationTests")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.conf.set("spark.sql.ansi.enabled", "true")
    yield session
    session.stop()


def validate(spark, rows, sent=None, **kwargs):
    received = spark.createDataFrame(rows, "id string, dt string, score double")
    submitted = spark.createDataFrame(
        sent if sent is not None else [(r[0], r[1]) for r in rows],
        "id string, dt string",
    )
    return validate_external_dataframe(
        submitted, received, ["id", "dt"], "dt", "score", **kwargs
    )


def test_matching_missingness_bands_and_density(spark, monkeypatch):
    frame_type = type(spark.range(0))

    def forbidden(*args, **kwargs):
        raise AssertionError("Raw conversion is forbidden")

    monkeypatch.setattr(frame_type, "toPandas", forbidden)
    original_collect = frame_type.collect

    def aggregate_only(frame):
        assert not {"id", "dt", "score"}.intersection(frame.columns)
        return original_collect(frame)

    monkeypatch.setattr(frame_type, "collect", aggregate_only)
    rows = [
        ("a", "2026-02-01", 1000.0),
        ("a", "2026-01-01", 0.0),
        ("b", "2026-01-01", 100.0),
        ("b", "2026-01-01", None),
        ("c", "2026-02-01", float("nan")),
        ("d", "2026-02-01", 1100.0),
        (None, "2026-02-01", -1.0),
        ("e", "bad", 500.0),
        ("f", None, 600.0),
    ]
    sent = [
        ("a", "2026-01-01"),
        ("a", "2026-01-01"),
        ("b", "2026-01-01"),
        ("z", "2026-01-01"),
        (None, "2026-02-01"),
    ]
    report = validate(spark, rows, sent)
    match = report["matching"]
    assert match["matched_distinct_keys"] == 2
    assert match["submitted"]["rows"] == 5
    assert match["received"]["rows"] == 9
    assert match["submitted"]["distinct_keys"] == 4
    assert match["received"]["distinct_keys"] == 8
    assert match["received"]["null_key_rows"] == 2
    assert match["received"]["duplicate_key_rows"] == 1
    assert match["missing_submitted_keys"] == 1
    assert match["unexpected_received_keys"] == 4
    assert match["submitted_match_percentage"] == pytest.approx(200 / 3)
    assert match["received_match_percentage"] == pytest.approx(100 / 3)
    assert report["scores"]["missing"] == 2
    assert report["scores"]["missing_percentage"] == pytest.approx(200 / 9)
    assert report["scores"]["below_range"] == report["scores"]["above_range"] == 1
    assert report["dates"] == {"missing_dates": 1, "invalid_dates": 1}
    assert report["baseline"] == "202601"
    assert [r["cohort"] for r in report["monthly_volume"]] == ["202601", "202602"]
    assert len(report["cutoffs"]) == 9
    for month in ("202601", "202602"):
        bands = [r for r in report["monthly_bands"] if r["cohort"] == month]
        assert len(bands) == 11
        assert sum(r["percentage"] for r in bands) == pytest.approx(100)
    feb = {
        r["band"]: r["count"]
        for r in report["monthly_bands"]
        if r["cohort"] == "202602"
    }
    assert feb["D10"] == 2  # Frozen January thresholds, not February quantiles.
    assert feb["D1"] == 1
    assert feb["Missing"] == 1
    if report["figures"]:
        assert len(report["figures"]["monthly_bands"].axes[0].containers) == 11
        assert (
            report["figures"]["histogram"]
            .axes[0]
            .get_ylabel()
            .startswith("Probability density")
        )
    hist = report["histogram"]
    assert [hist[i]["count"] for i in (0, 1, 9)] == [1, 1, 1]
    assert sum(r["density"] * 100 for r in hist) == pytest.approx(1)
    assert sum(r["percentage"] for r in hist) == pytest.approx(100)


def test_constant_baseline_and_explicit_selection(spark):
    rows = [
        ("a", "2026-01-01", 50.0),
        ("b", "2026-01-01", 50.0),
        ("a", "2026-02-01", 51.0),
        ("b", "2026-02-01", 49.0),
    ]
    report = validate(spark, rows, baseline="202601")
    assert report["cutoffs"] == [50.0] * 9
    assert [r["count"] for r in report["monthly_bands"][:11]] == [0, 2] + [0] * 9
    assert report["band_definitions"][1]["lower"] == float("-inf")
    assert report["band_definitions"][-1]["upper"] == float("inf")
    assert validate(spark, rows, baseline="202602")["baseline"] == "202602"


@pytest.mark.parametrize(
    "rows,baseline,message",
    [
        ([], None, "No valid monthly"),
        (
            [("a", "2026-01-01", None), ("a", "2026-02-01", 5.0)],
            None,
            "no valid scores",
        ),
        ([("a", "2026-01-01", 5.0)], "202512", "does not exist"),
        (
            [("a", "2026-01-01", 5.0), ("b", "2026-02-01", float("inf"))],
            None,
            "Invalid scores",
        ),
        ([("a", "2026-02-30", 5.0)], None, "No valid monthly"),
    ],
)
def test_unavailable_sections_preserve_metrics(spark, rows, baseline, message):
    report = validate(spark, rows, baseline=baseline)
    assert message in report["stability_status"]
    assert report["monthly_bands"] == []
    assert report["matching"]["received"]["rows"] == len(rows)
    assert len(report["histogram"]) == 10
    if not rows:
        assert report["scores"]["missing_percentage"] is None
        assert report["matching"]["received_match_percentage"] is None
        assert all(r["density"] is None for r in report["histogram"])


@pytest.mark.parametrize("value", [date(2026, 1, 1), datetime(2026, 1, 1, 12)])
def test_native_dates(spark, value):
    df = spark.createDataFrame([("a", value, 100.0)], ["id", "dt", "score"])
    report = validate_external_dataframe(df, df, ["id", "dt"], "dt", "score")
    assert report["baseline"] == "202601"


def test_input_validation(spark):
    df = spark.createDataFrame(
        [("a", "2026-01-01", "malformed")], ["id", "dt", "score"]
    )
    with pytest.raises(ValueError, match="numeric"):
        validate_external_dataframe(df, df, ["id"], "dt", "score")
    for keys in ([], "id", ["id", "id"]):
        with pytest.raises(ValueError, match="key_cols"):
            validate_external_dataframe(df, df, keys, "dt", "score")
    with pytest.raises(ValueError, match="Missing required"):
        validate_external_dataframe(df, df, ["absent"], "dt", "score")
    for baseline in ("202613", "2026-01", 202601):
        with pytest.raises(ValueError, match="baseline"):
            validate(spark, [], baseline=baseline)


def test_undated_invalid_score_and_decimal_boundaries(spark):
    from decimal import Decimal

    df = spark.createDataFrame(
        [
            ("a", "2026-01-01", Decimal("1000.000000000000000001")),
            ("b", "2026-01-01", Decimal("1000")),
        ],
        "count string, dt string, score decimal(25,18)",
    )
    report = validate_external_dataframe(df, df, ["count", "dt"], "dt", "score")
    assert report["scores"]["above_range"] == 1
    assert report["histogram"][-1]["count"] == 1
    assert report["matching"]["matched_distinct_keys"] == 2
    report = validate(spark, [("a", "2026-01-01", 50.0), ("b", None, float("inf"))])
    assert report["scores"]["invalid"] == 1
    assert report["stability_status"] == "available"


def test_notebook_section_order_and_headless_figures(monkeypatch):
    import io
    from types import ModuleType

    from brainmodelkit.pyspark.validation import _render_report

    displayed = []
    ipython = ModuleType("IPython")
    ipython.get_ipython = lambda: object()
    ipython.version_info = (9, 0)
    output = ModuleType("IPython.display")
    output.Markdown = lambda value: value
    output.HTML = lambda value: value
    output.display = displayed.append
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", output)
    report = dict(
        matching={"submitted": {"rows": 0}, "received": {"rows": 0}},
        scores={"missing": 0},
        dates={"missing_dates": 0, "invalid_dates": 0},
        monthly_volume=[],
        baseline=None,
        cutoffs=[],
        band_definitions=[],
        monthly_bands=[],
        histogram=[],
        stability_status="No valid monthly cohorts",
        warnings=[],
        figures={},
    )
    _render_report(report)
    titles = [
        value
        for value in displayed
        if isinstance(value, str) and value.startswith("###")
    ]
    assert titles == [
        "### Matching summary",
        "### Score missingness",
        "### Monthly received volume",
        "### Baseline selection and decile thresholds",
        "### Monthly score-band stability",
        "### Overall score density",
        "### Data-quality warnings",
    ]
    for figure in report["figures"].values():
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        assert buffer.getvalue().startswith(b"\x89PNG")
