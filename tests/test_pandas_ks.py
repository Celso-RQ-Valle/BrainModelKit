"""Tests for Pandas KS metrics."""

import pandas as pd
import pytest

from brainmodelkit.metrics.pandas import calculate_ks, ks


def test_calculate_ks_point_by_point() -> None:
    """Exact KS should compare cumulative score distributions."""
    target = pd.Series([1, 0, 1, 0])
    score = pd.Series([0.9, 0.8, 0.7, 0.6])

    assert calculate_ks(target, score) == pytest.approx(0.5)


def test_ks_returns_one_result_per_group() -> None:
    """Grouped KS should preserve group values in its result."""
    dataframe = pd.DataFrame(
        {
            "segment": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "target": [1, 0, 1, 0, 1, 1, 0, 0],
            "score": [0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.2, 0.1],
        }
    )

    result = ks("score", dataframe, "segment", "target")

    assert result.set_index("segment")["KS"].to_dict() == pytest.approx(
        {"A": 0.5, "B": 1.0}
    )


def test_calculate_ks_returns_nan_when_a_class_is_missing() -> None:
    """KS is undefined unless both target classes are present."""
    result = calculate_ks(pd.Series([1, 1]), pd.Series([0.9, 0.8]))

    assert pd.isna(result)
