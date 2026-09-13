"""Composable Pandas selectors: numerical contracts and reproducibility."""

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from brainmodelkit.feature_selection import SelectionResult
from brainmodelkit.feature_selection import pandas as fs


@pytest.fixture
def data():
    x, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        shuffle=False,
        random_state=42,
    )
    return pd.DataFrame(x, columns=list("abcd")).assign(target=y)


def test_quality_grouped_boundaries():
    df = pd.DataFrame(
        {
            "a": [1.0, 1, None, None],
            "b": [0.0] * 4,
            "month": ["Jan", "Jan", "Feb", "Feb"],
        }
    )
    result = fs.completeness(df, ["a", "b"], min_completeness=0.5, group_by="month")
    assert result.selected_features == ["a", "b"]
    assert result.feature_table[0]["missing_rate"] == 0.5
    assert len(result.grouped_table) == 4
    strict = fs.completeness(
        df,
        ["a", "b"],
        min_completeness=0.5,
        group_by=["month"],
        require_all_groups=True,
    )
    assert strict.rejected_features == ["a"]
    assert fs.variance_filter(df, ["a", "b"]).selected_features == []
    assert fs.cardinality(
        df, ["a", "b"], min_unique=1, max_unique=1, max_unique_ratio=0.25
    ).selected_features == ["a", "b"]
    assert (
        fs.completeness(
            df.assign(month=None), ["a"], group_by="month"
        ).grouped_table.shape[0]
        == 1
    )


def test_variance_population_and_all_null():
    df = pd.DataFrame({"a": [0.0, 2.0], "b": [np.nan, np.nan]})
    assert fs.variance_filter(df, ["a", "b"]).selected_features == ["a"]
    assert fs.variance_filter(df, ["a"], min_variance=1).selected_features == []
    assert fs.completeness(df, ["b"], min_completeness=0).selected_features == ["b"]
    assert fs.cardinality(df, ["b"], min_unique=0).feature_table[0]["unique_count"] == 0
    assert fs.completeness(df, df.columns).selected_features == ["a"]


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_correlation_determinism(method):
    df = pd.DataFrame({"a": [0.0, 1, 2, 3], "b": [0.0, -1, -2, -3], "c": [1.0] * 4})
    result = fs.correlation_filter(df, ["a", "b", "c"], threshold=0.99, method=method)
    assert result.selected_features == ["a", "c"]
    assert result.rejected_features == ["b"]
    assert result.correlation_pairs[0]["correlation"] == pytest.approx(-1)
    assert fs.correlation_filter(df, ["b", "a"], method=method).selected_features == [
        "b"
    ]


def test_supervised_statistics_and_rankings(data):
    for method in (fs.mutual_information, fs.anova):
        result = method(data, "target", list("abcd"), top_k=1)
        assert len(result.selected_features) == 1
        assert sorted(result.ranking.values()) == [1, 2, 3, 4]
    a = fs.mutual_information(data, "target", list("abcd"), random_state=8)
    b = fs.mutual_information(data, "target", list("abcd"), random_state=8)
    assert a.feature_table == b.feature_table
    counts = pd.DataFrame(
        {"signal": [0, 0, 1, 1] * 10, "constant": [0] * 40, "target": [0, 0, 1, 1] * 10}
    )
    result = fs.chi_square(counts, "target", ["signal", "constant"])
    assert result.selected_features == ["signal"]
    assert result.ranking["signal"] == 1
    for value in (-1, 0.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            fs.chi_square(counts.assign(signal=value), "target", ["signal"])


@pytest.mark.parametrize("binning", ["quantile", "uniform"])
def test_iv_missing_smoothing_and_formula(binning):
    df = pd.DataFrame(
        {
            "numeric": [0.0, 0, 1, 1, None, None],
            "category": ["a", "a", "b", "b", None, None],
            "empty": [None] * 6,
            "target": [0, 0, 1, 1, 0, 1],
        }
    )
    result = fs.information_value(
        df,
        "target",
        ["numeric", "category", "empty"],
        n_bins=2,
        binning=binning,
        min_iv=0.01,
    )
    assert result.selected_features == ["numeric", "category"]
    assert result.feature_table[-1]["iv"] == 0
    assert np.isfinite(result.bin_details.woe).all()
    for feature, table in result.bin_details.groupby("feature"):
        assert table.distribution_event.sum() == pytest.approx(1)
        assert table.distribution_non_event.sum() == pytest.approx(1)
        expected = (
            (table.distribution_event - table.distribution_non_event) * table.woe
        ).sum()
        assert next(
            r["iv"] for r in result.feature_table if r["feature"] == feature
        ) == pytest.approx(expected)
    assert result.bin_details.is_missing.any()
    # Real strings resembling missing markers must not collide with missing bins.
    separated = fs.information_value(
        df.assign(category=["missing", "missing", "b", "b", None, None]),
        "target",
        ["category"],
    )
    assert len(separated.bin_details) == 3


def test_importance_l1_and_permutation(data):
    features = list("abcd")
    model = LogisticRegression()
    result = fs.feature_importance_selection(
        data, "target", features, model=model, top_k=2
    )
    assert not hasattr(model, "coef_")
    assert len(result.selected_features) == 2
    assert all("coefficients" in r for r in result.feature_table)
    strongest = max(result.feature_table, key=lambda r: abs(r["importance"]))
    assert strongest["ranking"] == 1
    assert fs.l1_selection(data, "target", features, C=0.00001).selected_features == []
    perm = fs.permutation_importance_selection(
        result.model, data, "target", features, top_k=1, n_repeats=2
    )
    assert len(perm.selected_features) == 1
    assert all(r["importance_std"] >= 0 for r in perm.feature_table)
    with pytest.raises(ValueError, match="native importance"):
        fs.feature_importance_selection(
            data, "target", features, model=DummyClassifier()
        )


@pytest.mark.parametrize(
    "model", ["decision_tree", "random_forest", "gradient_boosting", "lightgbm"]
)
def test_importance_models(data, model):
    if model == "lightgbm":
        pytest.importorskip("lightgbm")
    params = {"verbosity": -1, "n_estimators": 5} if model == "lightgbm" else {}
    assert (
        len(
            fs.feature_importance_selection(
                data, "target", list("abcd"), model=model, model_params=params, top_k=2
            ).selected_features
        )
        == 2
    )


def test_cv_wrappers(data):
    features = list("abcd")
    a = fs.rfecv(data, "target", features, cv=3)
    b = fs.rfecv(data, "target", features, cv=3)
    assert a.feature_table == b.feature_table
    assert a.optimal_feature_count == len(a.selected_features)
    assert "mean_test_score" in a.cv_results
    for direction in ("forward", "backward"):
        assert (
            len(
                fs.sequential_selection(
                    data,
                    "target",
                    features,
                    direction=direction,
                    n_features_to_select=2,
                    cv=3,
                ).selected_features
            )
            == 2
        )


@pytest.mark.parametrize("method", ["feature_importance", "l1"])
def test_stability_reproducibility(data, method):
    kwargs = (
        {"top_k": 2, "model": "decision_tree"}
        if method == "feature_importance"
        else {"C": 0.1}
    )
    a = fs.stability_selection(
        data,
        "target",
        list("abcd"),
        method=method,
        n_iterations=3,
        selector_kwargs=kwargs,
    )
    b = fs.stability_selection(
        data,
        "target",
        list("abcd"),
        method=method,
        n_iterations=3,
        selector_kwargs=kwargs,
    )
    assert a.feature_table == b.feature_table
    assert all(
        r["selection_frequency"] == r["selection_count"] / 3 for r in a.feature_table
    )
    grouped = fs.stability_selection(
        data.assign(month=np.arange(len(data)) % 2),
        "target",
        list("abcd"),
        method=method,
        n_iterations=2,
        group_by="month",
        selector_kwargs=kwargs,
    )
    assert grouped.metadata["n_fits"] == 4


def test_optional_boruta_missing_and_contract(data, monkeypatch):
    original = importlib.import_module

    def missing(name):
        if name == "boruta":
            raise ImportError("missing")
        return original(name)

    monkeypatch.setattr(
        "brainmodelkit.feature_selection._pandas_wrappers.importlib.import_module",
        missing,
    )
    with pytest.raises(ImportError, match="pandas,boruta"):
        fs.boruta(data, "target", list("abcd"))

    class FakeBoruta:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, x, y):
            self.ranking_ = [1, 2, 3, 4]
            self.support_ = [True, False, False, False]
            self.support_weak_ = [False, True, False, False]

    monkeypatch.setattr(
        "brainmodelkit.feature_selection._pandas_wrappers.importlib.import_module",
        lambda name: SimpleNamespace(BorutaPy=FakeBoruta),
    )
    result = fs.boruta(data, "target", list("abcd"))
    assert result.selected_features == ["a"]
    assert [r["status"] for r in result.feature_table] == [
        "confirmed",
        "tentative",
        "rejected",
        "rejected",
    ]


def test_boruta_integration(data):
    pytest.importorskip("boruta")
    result = fs.boruta(data, "target", list("abcd"), n_estimators=20, max_iter=5)
    assert set(result.ranking) == set("abcd")


@pytest.mark.parametrize(
    "options",
    [
        {"feature_cols": []},
        {"feature_cols": ["missing"]},
        {"feature_cols": ["a", "a"]},
        {"min_completeness": 1.1},
        {"require_all_groups": True},
    ],
)
def test_quality_validation(data, options):
    kwargs = {"feature_cols": ["a"], **options}
    with pytest.raises(ValueError):
        fs.completeness(data, **kwargs)


@pytest.mark.parametrize("target", [[0] * 80, [np.nan] * 80, np.linspace(0, 1, 80)])
def test_invalid_targets(data, target):
    with pytest.raises(ValueError):
        fs.mutual_information(data.assign(target=target), "target", ["a"])


def test_invalid_numeric_and_options(data):
    for method in (fs.correlation_filter, fs.variance_filter):
        with pytest.raises(ValueError, match="numeric"):
            method(data.assign(a="text"), ["a"])
    with pytest.raises(ValueError, match="finite"):
        fs.correlation_filter(data.assign(a=np.nan), ["a"])
    with pytest.raises(ValueError):
        fs.information_value(data, "target", ["a"], smoothing=0)
    with pytest.raises(ValueError):
        fs.mutual_information(data, "target", ["a"], top_k=2)
    with pytest.raises(ValueError):
        fs.cardinality(data, ["a"], min_unique=3, max_unique=2)
    with pytest.raises(ValueError):
        fs.completeness(data.iloc[:0], ["a"])
    assert isinstance(fs.completeness(data, ["a"]), SelectionResult)


def test_pandas_import_without_spark():
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyspark" or fullname.startswith("pyspark."):
            raise AssertionError("Pandas imported Spark")
sys.meta_path.insert(0, Block())
import pandas as pd
from brainmodelkit.feature_selection.pandas import completeness
assert completeness(pd.DataFrame({"x": [1]}), ["x"]).selected_features == ["x"]
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
