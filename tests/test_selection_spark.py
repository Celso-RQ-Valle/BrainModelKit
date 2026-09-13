"""Native Spark selector contracts; conversion is forbidden for every test."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from brainmodelkit.feature_selection import pyspark as fs


@pytest.fixture(scope="module")
def spark():
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    session = (
        SparkSession.builder.master("local[1]")
        .appName("SelectionTests")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(autouse=True)
def forbid_conversion(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Spark selection must not convert data to Pandas")

    monkeypatch.setattr(DataFrame, "toPandas", fail)
    # Spark 4 delegates to a classic subclass which overrides the base method.
    try:
        from pyspark.sql.classic.dataframe import DataFrame as ClassicDataFrame
    except ImportError:
        pass
    else:
        monkeypatch.setattr(ClassicDataFrame, "toPandas", fail)


def test_quality_and_grouped_deterioration(spark):
    df = spark.createDataFrame(
        [
            (1.0, 0.0, "Jan"),
            (1.0, 0.0, "Jan"),
            (None, 0.0, "Feb"),
            (float("nan"), 0.0, "Feb"),
        ],
        "a double, b double, month string",
    )
    result = fs.completeness(df, ["a", "b"], min_completeness=0.5, group_by="month")
    assert result.selected_features == ["a", "b"]
    assert isinstance(result.grouped_table, DataFrame)
    assert result.grouped_table.count() == 4
    assert result.feature_table[0]["missing_rate"] == 0.5
    strict = fs.completeness(
        df,
        ["a", "b"],
        min_completeness=0.5,
        group_by=["month"],
        require_all_groups=True,
    )
    assert strict.rejected_features == ["a"]
    assert fs.variance_filter(df, ["a", "b"]).selected_features == []
    result = fs.cardinality(
        df, ["a", "b"], min_unique=1, max_unique=1, max_unique_ratio=0.25
    )
    assert result.selected_features == ["a", "b"]
    empty = df.withColumn("a", F.lit(None).cast("double"))
    assert fs.completeness(empty, ["a"]).rejected_features == ["a"]
    assert fs.variance_filter(empty, ["a"]).rejected_features == ["a"]


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_correlation_native(spark, method):
    df = spark.createDataFrame(
        [(0.0, 0.0), (1.0, -1.0), (2.0, -2.0)], "a double, b double"
    )
    result = fs.correlation_filter(df, ["a", "b"], method=method)
    assert result.selected_features == ["a"]
    assert result.correlation_pairs[0]["correlation"] == pytest.approx(-1)
    with pytest.raises(ValueError, match="max_features"):
        fs.correlation_filter(df, ["a", "b"], max_features=1)


def test_chi_square_native(spark):
    df = spark.createDataFrame([(0, 0, 0), (1, 0, 1)] * 10, "a int, b int, target int")
    result = fs.chi_square(df, "target", ["a", "b"])
    assert result.selected_features == ["a"]
    assert result.ranking["a"] == 1
    for value in (-1.0, 0.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            fs.chi_square(df.withColumn("a", F.lit(value)), "target", ["a"])


@pytest.mark.parametrize("binning", ["quantile", "uniform"])
def test_iv_native_missing_and_smoothing(spark, binning):
    df = spark.createDataFrame(
        [
            (0.0, "a", 0),
            (0.0, "a", 0),
            (1.0, "b", 1),
            (1.0, "b", 1),
            (None, None, 0),
            (None, None, 1),
        ],
        "a double, category string, target int",
    )
    result = fs.information_value(
        df, "target", ["a", "category"], n_bins=2, min_iv=0.01, binning=binning
    )
    assert result.selected_features == ["a", "category"]
    assert isinstance(result.bin_details, DataFrame)
    summaries = (
        result.bin_details.groupBy("feature")
        .agg(
            F.sum("distribution_event").alias("de"),
            F.sum("distribution_non_event").alias("dn"),
            F.sum("iv_component").alias("iv"),
        )
        .collect()
    )
    for row in summaries:
        assert row.de == pytest.approx(1)
        assert row.dn == pytest.approx(1)
        assert next(
            r["iv"] for r in result.feature_table if r["feature"] == row.feature
        ) == pytest.approx(row.iv)
    assert result.bin_details.filter("is_missing").count() == 2
    constant = fs.information_value(
        df.withColumn("empty", F.lit(None)), "target", ["empty"]
    )
    assert constant.feature_table[0]["iv"] == 0


@pytest.mark.parametrize(
    "model",
    ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting"],
)
def test_model_importance_native(spark, model):
    df = spark.createDataFrame(
        [(-2.0, 0.0, 0), (-1.0, 0.0, 0), (1.0, 0.0, 1), (2.0, 0.0, 1)] * 2,
        "a double, b double, target int",
    )
    params = (
        {"numTrees": 3}
        if model == "random_forest"
        else {"maxIter": 3}
        if model in ("gradient_boosting", "logistic_regression")
        else {}
    )
    result = fs.feature_importance_selection(
        df, "target", ["a", "b"], model=model, model_params=params, top_k=1
    )
    assert result.selected_features == ["a"]
    if model == "logistic_regression":
        assert "coefficients" in result.feature_table[0]


def test_l1_and_rfe_native(spark, tmp_path, monkeypatch):
    df = spark.createDataFrame(
        [(-2.0, 0.0, 0), (-1.0, 0.0, 0), (1.0, 0.0, 1), (2.0, 0.0, 1)],
        "a double, b double, target int",
    )
    assert (
        fs.l1_selection(df, "target", ["a", "b"], reg_param=100).selected_features == []
    )
    # Native writer behavior is covered in training tests; Windows lacks winutils.
    monkeypatch.setattr(
        "brainmodelkit.training._common._save_spark_model", lambda *a, **kw: None
    )
    result = fs.rfe(
        "existing",
        "target",
        df,
        df,
        ["a", "b"],
        n_feat_final=1,
        model_params={"maxIter": 3},
        output_dir=tmp_path,
        n_tiles=2,
    )
    assert result.selected_features == ["a"]
    assert [r["n_features"] for r in result.history] == [2, 1]
    assert result.history[0]["removed_features"] == ["b"]


def test_invalid_inputs(spark):
    df = spark.createDataFrame([(1.0, 0), (2.0, 0)], "a double, target int")
    with pytest.raises(ValueError, match="both binary"):
        fs.information_value(df, "target", ["a"])
    with pytest.raises(ValueError, match="Missing"):
        fs.completeness(df, ["absent"])
    with pytest.raises(ValueError, match="finite"):
        fs.correlation_filter(df.withColumn("a", F.lit(float("nan"))), ["a"])
    with pytest.raises(ValueError, match="at least one"):
        fs.completeness(df.limit(0), ["a"])
    assert fs.variance_filter(df, ["a"], min_variance=0.25).selected_features == []


def test_spark_namespace_isolated():
    available = {
        "mutual_information",
        "anova",
        "rfecv",
        "sequential_selection",
        "stability_selection",
        "permutation_importance_selection",
        "boruta",
    }
    assert all(hasattr(fs, name) for name in available)
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.startswith("sklearn")
            or fullname.startswith("brainmodelkit.feature_selection._pandas")
            or fullname == "brainmodelkit.feature_selection.pandas"):
            raise AssertionError("Spark selector imported Pandas implementation")
sys.meta_path.insert(0, Block())
from brainmodelkit.feature_selection.pyspark import completeness, information_value, rfe
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def test_new_univariate_statistics(spark):
    import math

    from scipy.stats import f_oneway

    rows = [(float(i % 4), float(i % 2), 1.0, i % 2) for i in range(40)]
    df = spark.createDataFrame(
        rows, "x double, signal double, constant double, target int"
    )
    result = fs.mutual_information(
        df, "target", ["signal", "constant"], discrete_features=True
    )
    assert result.feature_table[0]["mutual_information"] == pytest.approx(math.log(2))
    assert result.feature_table[1]["mutual_information"] == 0
    binned = fs.mutual_information(df, "target", ["x"], n_bins=4, relative_error=0)
    assert binned.metadata["estimator"] == "binned"
    result = fs.anova(df, "target", ["x", "signal", "constant"])
    expected = f_oneway(
        [r[0] for r in rows if r[-1] == 0], [r[0] for r in rows if r[-1] == 1]
    )
    assert result.feature_table[0]["f_statistic"] == pytest.approx(expected.statistic)
    assert result.feature_table[0]["p_value"] == pytest.approx(expected.pvalue)
    assert result.feature_table[1]["f_statistic"] == float("inf")
    assert result.feature_table[2]["selected"] is False
    with pytest.raises(ValueError, match="mask"):
        fs.mutual_information(df, "target", ["x"], discrete_features=[])


@pytest.fixture
def search_data(spark):
    before = spark.sparkContext._jsc.getPersistentRDDs().size()
    yield spark.createDataFrame(
        [(float(i % 2), 0.0, i % 2, (i // 2) % 2) for i in range(40)],
        "signal double, constant double, target int, fold int",
    )
    assert spark.sparkContext._jsc.getPersistentRDDs().size() == before


def test_distributed_permutation(search_data):
    from brainmodelkit.feature_selection._spark_search import indexed, shuffled

    with indexed(search_data) as frame:
        permuted = shuffled(frame, "signal", 42)
        assert permuted.count() == search_data.count()
        assert (
            permuted.groupBy("signal").count().orderBy("signal").collect()
            == search_data.groupBy("signal").count().orderBy("signal").collect()
        )
    model = fs.feature_importance_selection(
        search_data, "target", ["signal", "constant"], model="decision_tree"
    ).model
    result = fs.permutation_importance_selection(
        model, search_data, "target", ["signal", "constant"], n_repeats=1
    )
    assert result.feature_table[0]["importance_mean"] > 0
    assert result.feature_table[1]["importance_mean"] == 0


def test_spark_rfecv(search_data):
    result = fs.rfecv(
        search_data,
        "target",
        ["signal", "constant"],
        model="decision_tree",
        cv=2,
        fold_col="fold",
    )
    assert result.selected_features == ["signal"]
    assert result.cv_results["n_features"] == [1, 2]
    assert result.model.numFeatures == 1


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_spark_sequential(search_data, direction):
    result = fs.sequential_selection(
        search_data,
        "target",
        ["signal", "constant"],
        model="decision_tree",
        cv=2,
        fold_col="fold",
        direction=direction,
    )
    assert result.selected_features == ["signal"]
    assert len(result.history[0]["candidates"]) == 2


def test_spark_stability(search_data):
    result = fs.stability_selection(
        search_data,
        "target",
        ["signal", "constant"],
        n_iterations=2,
        group_by="fold",
        selector_kwargs={"model": "decision_tree", "top_k": 1},
    )
    assert result.selected_features == ["signal"]
    assert result.metadata["n_fits"] == 4
    assert result.feature_table[0]["selection_frequency"] == 1


def test_spark_boruta(search_data):
    result = fs.boruta(
        search_data,
        "target",
        ["signal", "constant"],
        n_estimators=3,
        max_iter=1,
        model_params={"maxDepth": 2},
    )
    assert result.selected_features == []
    assert all(r["status"] == "tentative" for r in result.feature_table)
    assert result.history[0]["shadow_threshold"] >= 0
    assert result.feature_table[1]["hits"] == 0


def test_shadow_hypothesis_decisions():
    from brainmodelkit.feature_selection._spark_robustness import decisions

    pa, pr, accept, reject = decisions([20, 0, 10], 20, 0.05, True, 3)
    assert accept.tolist() == [True, False, False]
    assert reject.tolist() == [False, True, False]
    assert pa[0] == pytest.approx(2**-20)
    assert pr[1] == pytest.approx(2**-20)


def test_owned_cache_releases_on_error(search_data):
    from brainmodelkit.feature_selection._spark_search import owned_cache

    search_data.persist()
    search_data.count()
    context = search_data.sparkSession.sparkContext
    before = context._jsc.getPersistentRDDs().size()
    try:
        with (
            pytest.raises(RuntimeError, match="test failure"),
            owned_cache(search_data) as cached,
        ):
            assert cached.count() == 40
            raise RuntimeError("test failure")
        assert search_data.is_cached
        assert context._jsc.getPersistentRDDs().size() == before
    finally:
        search_data.unpersist(blocking=True)
