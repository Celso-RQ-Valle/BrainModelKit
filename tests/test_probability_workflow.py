"""Probability semantics, fixed scales, fitting, and portable configurations."""

import numpy as np
import pandas as pd
import pytest

from brainmodelkit.calibration.pandas import (
    LogisticCalibrator,
    calibration_diagnostics,
    compare_calibration,
)
from brainmodelkit.scoring.pandas import ScoreAligner


def test_manual_semantics_and_no_mutation():
    frame = pd.DataFrame(
        {"id": [1, 1, 2], "part": [0, 1, 0], "score": [0.0, 0.2, 1.0]}, index=[4, 4, 8]
    )
    original = frame.copy(deep=True)
    cal = LogisticCalibrator(0, 1)
    result = cal.transform(frame, event_col="event", non_event_col="non_event")
    np.testing.assert_allclose(
        result.event, np.clip(frame.score, cal.epsilon, 1 - cal.epsilon)
    )
    np.testing.assert_allclose(result.event + result.non_event, 1)
    pd.testing.assert_frame_equal(frame, original)
    assert result.index.equals(frame.index)
    complement = LogisticCalibrator(
        0, 1, input_probability="non_event", output_probability="non_event"
    )
    other = complement.transform(frame.assign(score=1 - frame.score))
    np.testing.assert_allclose(other.calibrated_probability, result.non_event)
    assert complement.slope == 1 and complement.mode == "manual"
    for slope, ranking in [(1, "preserves"), (-1, "reverses"), (0, "collapses")]:
        assert LogisticCalibrator(0, slope).ranking == ranking


def test_probability_vs_logit_raw_and_extremes():
    p = pd.Series([0.2, 0.7])
    for space, x in [("probability", p), ("logit", np.log(p / (1 - p)))]:
        cal = LogisticCalibrator(-0.3, 1.4, input_space=space)
        np.testing.assert_allclose(
            cal.predict_event(p), 1 / (1 + np.exp(-(-0.3 + 1.4 * x)))
        )
    raw = LogisticCalibrator(0, 1, input_space="raw")
    np.testing.assert_array_equal(
        raw.predict_event(pd.Series([-1e300, 0, 1e300])), [0, 0.5, 1]
    )
    with pytest.raises(ValueError, match="Raw margins"):
        LogisticCalibrator(0, 1, input_space="raw", input_probability="event")


@pytest.mark.parametrize("values", [[-0.2], [1.3], [np.inf], [np.nan]])
def test_invalid_values(values):
    frame = pd.DataFrame({"score": values})
    for transform in [LogisticCalibrator(0, 1), ScoreAligner()]:
        with pytest.raises(ValueError):
            transform.transform(frame)
        transform.invalid_policy = "null"
        assert transform.transform(frame).iloc[:, -1].isna().all()


def test_validation_collisions_and_serialization(tmp_path):
    for args in [{"intercept": 1}, {"intercept": np.inf, "slope": 1}, {"epsilon": 0.5}]:
        with pytest.raises(ValueError):
            LogisticCalibrator(**args)
    with pytest.raises(ValueError):
        ScoreAligner(score_min=3, score_max=3)
    with pytest.raises(TypeError):
        LogisticCalibrator(0, 1).transform(pd.DataFrame({"score": ["0.2"]}))
    frame = pd.DataFrame({"score": [0.2]})
    with pytest.raises(ValueError, match="collision"):
        ScoreAligner().transform(frame, output_col="score")
    assert (
        ScoreAligner()
        .transform(frame, output_col="score", overwrite=True)
        .score.iloc[0]
        == 800
    )
    for config in [
        LogisticCalibrator(0.2, 0.8, feature_order=["x"]),
        ScoreAligner(200, 900, decimals=2),
    ]:
        path = tmp_path / (type(config).__name__ + ".json")
        config.save(path)
        assert type(config).load(path).to_dict() == config.to_dict()
        with pytest.raises(ValueError):
            type(config).from_dict({**config.to_dict(), "schema_version": 99})


def controlled_data():
    x = np.repeat(np.linspace(-2, 2, 15), 2)
    y = np.tile([0, 1], 15)
    p = 1 / (1 + np.exp(-(-0.4 + 1.3 * x)))
    return pd.DataFrame(
        {
            "score": 1 / (1 + np.exp(-x)),
            "target": y,
            "weight": np.where(y == 1, p, 1 - p),
        }
    )


def test_controlled_fit_and_failures():
    df = controlled_data()
    cal = LogisticCalibrator().fit(df, weight_col="weight")
    np.testing.assert_allclose([cal.intercept, cal.slope], [-0.4, 1.3], atol=1e-6)
    assert cal.diagnostics_["converged"] and cal.mode == "fitted"
    with pytest.raises(ValueError, match="Manual"):
        LogisticCalibrator(0, 1).fit(df)
    with pytest.raises(ValueError, match="Constant"):
        LogisticCalibrator().fit(df.assign(score=0.5))
    with pytest.raises(ValueError, match="Separated"):
        LogisticCalibrator().fit(df.assign(score=df.target.astype(float)))
    with pytest.raises(ValueError, match="Binary"):
        LogisticCalibrator().fit(df.assign(target=1))
    with pytest.raises(ValueError, match="Weights"):
        LogisticCalibrator().fit(df.assign(weight=-1), weight_col="weight")
    with pytest.raises(RuntimeError, match="converge"):
        LogisticCalibrator().fit(df, weight_col="weight", max_iter=1)


@pytest.mark.parametrize(
    "direction", ["lower_event_probability", "higher_event_probability"]
)
def test_score_endpoints_inverse_batch_invariance(direction):
    df = pd.DataFrame({"score": [0.0, 0.2, 1.0]})
    align = ScoreAligner(higher_score_means=direction)
    result = align.transform(df)
    expected = [1000, 800, 0] if direction.startswith("lower") else [0, 200, 1000]
    np.testing.assert_allclose(result.business_score, expected)
    np.testing.assert_allclose(
        align.inverse_transform(result).event_probability, df.score
    )
    assert (
        align.transform(df.iloc[[1]]).business_score.iloc[0]
        == result.business_score.iloc[1]
    )
    align.input_probability = "non_event"
    np.testing.assert_allclose(
        align.transform(df.assign(score=1 - df.score)).business_score, expected
    )
    custom = ScoreAligner(300, 850, higher_score_means=direction)
    assert custom.transform(df).business_score.between(300, 850).all()


def test_half_even_and_diagnostics():
    df = pd.DataFrame({"score": [0.125, 0.375, 0.625, 0.875], "target": [0, 1, 0, 1]})
    result = ScoreAligner(
        0, 4, higher_score_means="higher_event_probability", decimals=0
    ).transform(df)
    np.testing.assert_array_equal(result.business_score, [0, 2, 2, 4])
    np.testing.assert_allclose(result.business_score_unrounded, [0.5, 1.5, 2.5, 3.5])
    diagnostic = calibration_diagnostics(df, n_bins=2)
    assert diagnostic["count"] == 4
    assert diagnostic["brier_score"] == pytest.approx(
        np.mean((df.score - df.target) ** 2)
    )
    assert diagnostic["reliability"]["count"].tolist() == [2, 2]
    transformed = LogisticCalibrator(0, 1).transform(df)
    compared = compare_calibration(transformed, "score", "calibrated_probability")
    assert compared["before"]["log_loss"] == pytest.approx(
        compared["after"]["log_loss"]
    )
