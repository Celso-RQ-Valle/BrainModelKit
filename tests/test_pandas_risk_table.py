"""Risk table counts, boundaries, and validation."""

import pandas as pd
import pytest

from brainmodelkit.metrics.pandas import risk_table


def test_risk_table_ranges_counts_and_rates() -> None:
    df = pd.DataFrame({"score": range(1, 11), "target": [0] * 5 + [1] * 5})
    original = df.copy(deep=True)
    result = risk_table(df=df, n_tiles=3)
    assert result.to_dict("list") == {
        "n_tile": [1, 2, 3],
        "minimum_range": [7, 4, 1],
        "maximum_range": [10, 6, 3],
        "total_volume": [4, 3, 3],
        "total_events": [4, 1, 0],
        "total_non_events": [0, 2, 3],
        "event_rate": [1, 1 / 3, 0],
    }
    pd.testing.assert_frame_equal(df, original)
    assert risk_table(df=df, n_tiles=3, ascending=True)["minimum_range"].tolist() == [
        1,
        5,
        8,
    ]


def test_risk_table_missing_ties_and_small_inputs() -> None:
    df = pd.DataFrame({"prediction": [5, 5, 5, None, 1], "label": [1, 0, 1, 0, None]})
    result = risk_table("prediction", df, "label", n_tiles=2)
    assert result["total_volume"].tolist() == [2, 1]
    assert result["total_events"].tolist() == [1, 1]
    assert result["minimum_range"].tolist() == [5, 5]
    assert result["maximum_range"].tolist() == [5, 5]
    assert len(risk_table("prediction", df, "label")) == 3
    assert risk_table("prediction", df, "label", n_tiles=1).iloc[0][
        "event_rate"
    ] == pytest.approx(2 / 3)
    assert risk_table("prediction", df.iloc[:0], "label").empty


def test_risk_table_validation() -> None:
    df = pd.DataFrame({"score": [1, 2], "target": [0, 1]})
    with pytest.raises(TypeError):
        risk_table()
    for value in [True, 2.5, "10"]:
        with pytest.raises(TypeError):
            risk_table(df=df, n_tiles=value)
    with pytest.raises(ValueError):
        risk_table(df=df, n_tiles=0)
    with pytest.raises(TypeError):
        risk_table(df=df, ascending="yes")
    with pytest.raises(KeyError):
        risk_table(df=df, score_column="missing")
    with pytest.raises(ValueError, match="binary"):
        risk_table(df=df.assign(target=[0, 2]))
    with pytest.raises(ValueError, match="finite"):
        risk_table(df=df.assign(score=[1, float("inf")]))
    with pytest.raises(TypeError, match="numeric"):
        risk_table(df=df.assign(score=["a", "b"]))
