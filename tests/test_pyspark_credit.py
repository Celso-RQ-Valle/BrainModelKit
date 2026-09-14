"""Tests for synthetic PySpark credit data generation."""

from typing import Any

import pytest

from brainmodelkit.pyspark import simulate_credit_data


class FakeSparkSession:
    """Capture DataFrame creation without starting a Java process."""

    def createDataFrame(
        self,
        rows: list[tuple[Any, ...]],
        columns: list[str],
    ) -> tuple[list[tuple[Any, ...]], list[str]]:
        return rows, columns


def test_simulate_credit_data_uses_twenty_features_by_default() -> None:
    """The public generator should create 20 feature columns by default."""
    dataframe, feature_columns = simulate_credit_data(
        FakeSparkSession(),  # type: ignore[arg-type]
        row_count=5,
    )
    rows, columns = dataframe

    assert len(rows) == 5
    assert len(feature_columns) == 20
    assert feature_columns[0] == "feature_01"
    assert feature_columns[-1] == "feature_20"
    assert columns[-20:] == feature_columns
    assert "company_size" in columns
    assert {row[5] for row in rows} <= {
        "Small",
        "Medium",
        "Large",
        "Very Large",
    }
    assert all(len(row) == len(columns) for row in rows)


def test_simulate_credit_data_accepts_a_custom_feature_count() -> None:
    """Callers should be able to control the generated feature count."""
    _, feature_columns = simulate_credit_data(
        FakeSparkSession(),  # type: ignore[arg-type]
        row_count=2,
        feature_count=3,
    )

    assert feature_columns == ["feature_01", "feature_02", "feature_03"]


@pytest.mark.parametrize("parameter", ["row_count", "feature_count", "max_days"])
def test_simulate_credit_data_rejects_non_positive_values(parameter: str) -> None:
    """Invalid size controls should fail with a useful error."""
    arguments = {parameter: 0}

    with pytest.raises(ValueError, match=f"{parameter} must be at least 1"):
        simulate_credit_data(FakeSparkSession(), **arguments)  # type: ignore[arg-type]


def test_simulate_credit_data_has_a_small_noisy_signal_subset() -> None:
    dataframe, feature_columns = simulate_credit_data(
        FakeSparkSession(), row_count=2000, feature_count=12, seed=7
    )
    rows, _ = dataframe
    informative_count = max(1, int(12 / 6))
    correlations = []
    for index in range(12):
        values = [row[6 + index] for row in rows]
        targets = [row[3] for row in rows]
        mean_x = sum(values) / len(values)
        mean_y = sum(targets) / len(targets)
        covariance = sum(
            (x - mean_x) * (y - mean_y) for x, y in zip(values, targets, strict=True)
        )
        variance_x = sum((x - mean_x) ** 2 for x in values)
        variance_y = sum((y - mean_y) ** 2 for y in targets)
        correlations.append(abs(covariance / (variance_x * variance_y) ** 0.5))
    assert len(feature_columns) == 12
    assert sum(value > 0.10 for value in correlations) == informative_count
    assert max(correlations) < 0.6


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("informative_fraction", -0.1),
        ("informative_fraction", 1.1),
        ("signal_strength", 0),
    ],
)
def test_simulate_credit_data_rejects_invalid_signal_controls(
    parameter: str, value: float
) -> None:
    with pytest.raises(ValueError, match=parameter):
        simulate_credit_data(FakeSparkSession(), **{parameter: value})
