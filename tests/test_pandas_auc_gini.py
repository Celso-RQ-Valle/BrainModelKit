"""Behavioral tests for Pandas ROC AUC and Gini."""

import pandas as pd
import pytest

from brainmodelkit.metrics.pandas import auc_gini


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([0.9, 0.8, 0.2, 0.1], 1.0),
        ([0.1, 0.2, 0.8, 0.9], 0.0),
        ([0.5, 0.5, 0.5, 0.5], 0.5),
        ([0.9, 0.2, 0.8, 0.1], 0.75),
    ],
)
def test_overall_defaults(scores: list[float], expected: float) -> None:
    result = auc_gini(df=pd.DataFrame({"target": [1, 1, 0, 0], "score": scores}))
    assert list(result.columns) == ["auc", "gini"]
    assert result.iloc[0].to_dict() == pytest.approx(
        {"auc": expected, "gini": 2 * expected - 1}
    )


def test_custom_columns_and_pairwise_missing_values() -> None:
    df = pd.DataFrame({"label": [1, 0, None, 1], "prediction": [0.9, 0.1, 1, None]})
    result = auc_gini("prediction", df, target_column="label")
    assert result.iloc[0].to_dict() == {"auc": 1.0, "gini": 1.0}


@pytest.mark.parametrize("targets", [[], [1, 1], [None, None]])
def test_undefined_metrics(targets: list) -> None:
    result = auc_gini(
        df=pd.DataFrame({"target": targets, "score": [0.5] * len(targets)})
    )
    assert result.shape == (1, 2)
    assert result.isna().all().all()


@pytest.mark.parametrize("group_by", ["segment", ["segment"], ["segment", "region"]])
def test_grouped_metrics_preserve_missing_keys(group_by: str | list[str]) -> None:
    df = pd.DataFrame(
        {
            "segment": ["B", "B", "A", None, None],
            "region": ["X"] * 5,
            "target": [1, 0, 1, 1, 0],
            "score": [0.9, 0.1, 0.8, 0.1, 0.9],
        }
    )
    result = auc_gini(df=df, group_by=group_by)
    assert len(result) == 3
    assert result.iloc[0]["segment"] == "B"
    assert result.iloc[0]["auc"] == 1.0
    assert pd.isna(result.iloc[1]["auc"])
    assert pd.isna(result.iloc[2]["segment"])
    assert result.iloc[2]["gini"] == -1.0


def test_empty_grouped_result_has_expected_columns() -> None:
    df = pd.DataFrame(columns=["segment", "score", "target"])
    result = auc_gini(df=df, group_by="segment")
    assert result.empty
    assert list(result.columns) == ["segment", "auc", "gini"]


def test_categorical_groups_only_include_observed_values() -> None:
    df = pd.DataFrame(
        {
            "segment": pd.Categorical(["A", "A"], categories=["A", "B"]),
            "score": [0.9, 0.1],
            "target": [1, 0],
        }
    )
    assert auc_gini(df=df, group_by="segment")["segment"].tolist() == ["A"]


@pytest.mark.parametrize(
    "group_by", [[], 123, [1], ("segment",), ["segment", "segment"], "auc"]
)
def test_invalid_grouping(group_by: object) -> None:
    with pytest.raises(ValueError):
        auc_gini(df=pd.DataFrame(), group_by=group_by)


def test_invalid_dataframe_and_missing_columns() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        auc_gini()
    with pytest.raises(KeyError, match="target"):
        auc_gini(df=pd.DataFrame({"score": [0.5]}))


def test_invalid_multiclass_target_raises() -> None:
    with pytest.raises(ValueError):
        auc_gini(df=pd.DataFrame({"score": [0.1, 0.5, 0.9], "target": [0, 1, 2]}))
