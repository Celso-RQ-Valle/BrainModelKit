"""Centralized Pandas cross-validation contracts."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier

from brainmodelkit.model_selection.pandas import cross_validate


@pytest.fixture
def data():
    x, y = make_classification(
        n_samples=60, n_features=3, n_informative=2, n_redundant=0, random_state=4
    )
    frame = pd.DataFrame(x, columns=["a", "b", "c"])
    frame["target"] = y
    return frame


def test_cv_is_deterministic_and_returns_fold_summary(data):
    kwargs = dict(
        df=data,
        target_col="target",
        feature_cols=["a", "b", "c"],
        model=DummyClassifier(strategy="prior"),
        cv=3,
    )
    first = cross_validate(**kwargs)
    second = cross_validate(**kwargs)
    pd.testing.assert_frame_equal(first.fold_metrics, second.fold_metrics)
    assert list(first.summary["metric"]) == ["KS", "AUC", "GINI"]
    assert first.models is None


def test_group_and_time_strategies(data):
    data["group"] = np.repeat(np.arange(15), 4)
    result = cross_validate(
        data,
        "target",
        ["a", "b", "c"],
        DummyClassifier(),
        cv=3,
        strategy="group",
        group_col="group",
    )
    assert len(result.fold_metrics) == 3
    data["date"] = np.arange(len(data))
    ordered = cross_validate(
        data,
        "target",
        ["a", "b", "c"],
        DummyClassifier(),
        cv=3,
        strategy="time_series",
        date_col="date",
    )
    assert len(ordered.fold_metrics) == 3


def test_invalid_columns_and_target(data):
    with pytest.raises(ValueError, match="Missing"):
        cross_validate(data, "target", ["missing"], DummyClassifier())
    with pytest.raises(ValueError, match="exclude"):
        cross_validate(data, "target", ["target"], DummyClassifier())
    with pytest.raises(ValueError, match="binary"):
        cross_validate(data.assign(target=2), "target", ["a"], DummyClassifier())
