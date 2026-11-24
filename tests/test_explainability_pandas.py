"""Attribution units, complete callable reconstruction, schemas and reporting."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from brainmodelkit._probability import CapabilityError
from brainmodelkit.calibration.pandas import LogisticCalibrator
from brainmodelkit.explainability.pandas import ModelExplainer, PredictionPipeline
from brainmodelkit.scoring.pandas import ScoreAligner
from brainmodelkit.training import TrainingResult


@pytest.fixture
def sample():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(100, 3)), columns=["income", "debt", "age"])
    y = (x.income - x.debt + rng.normal(size=100) > 0).astype(int)
    return x, y


def assert_reconstruction(result):
    np.testing.assert_allclose(
        result.base_values + result.contributions.sum(axis=1),
        result.predictions,
        atol=1e-6,
    )
    assert result.to_frame().reconstruction_ok.all()


def test_logistic_raw_event_class_and_training_result(sample):
    x, y = sample
    model = LogisticRegression().fit(x, y.map({0: "event", 1: "other"}))
    trained = TrainingResult(model, None, None, {}, [], None)
    explainer = ModelExplainer(
        trained, x.iloc[:20], event_label="event", key_cols=["client", "month"]
    )
    rows = x.iloc[30:33].assign(client=[1, 1, 2], month=[1, 2, 1])
    result = explainer.explain(rows)
    assert_reconstruction(result)
    np.testing.assert_allclose(
        result.predictions, -model.decision_function(x.iloc[30:33])
    )
    assert result.keys.client.tolist() == [1, 1, 2]
    assert result.metadata["event_label"] == "event"
    assert result.contributions.shape == (3, 3)


def test_complete_pipeline_probability_and_score(sample):
    x, y = sample
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, y)
    calibration = LogisticCalibrator(-0.4, 1.7, output_probability="non_event")
    aligner = ScoreAligner(300, 850, decimals=0, input_probability="non_event")
    workflow = PredictionPipeline(
        model, list(x), calibrator=calibration, score_aligner=aligner
    )
    kwargs = dict(method="permutation", max_background=12, max_evals=35, seed=9)
    results = {}
    for target in [
        "calibrated_event_probability",
        "calibrated_non_event_probability",
        "business_score",
    ]:
        results[target] = ModelExplainer(
            workflow, x.iloc[:20], target=target, **kwargs
        ).explain(x.iloc[30:32])
        assert_reconstruction(results[target])
    event, non_event, score = (results[name] for name in results)
    p = model.predict_proba(x.iloc[30:32])[:, 1]
    np.testing.assert_allclose(event.predictions, calibration.predict_event(p))
    np.testing.assert_allclose(non_event.contributions, -event.contributions)
    np.testing.assert_allclose(non_event.base_values, 1 - event.base_values)
    np.testing.assert_allclose(
        score.contributions, -550 * event.contributions, atol=1e-10
    )
    np.testing.assert_allclose(score.base_values, 850 - 550 * event.base_values)
    assert "rounding_difference" in score.to_frame()
    original = x.iloc[30:32].copy()
    scored = workflow.transform(original)
    np.testing.assert_allclose(scored.business_score_unrounded, score.predictions)
    pd.testing.assert_frame_equal(original, x.iloc[30:32])
    with pytest.raises(CapabilityError, match="calibration"):
        ModelExplainer(
            workflow, x.iloc[:20], target="calibrated_event_probability"
        ).explain(x.iloc[:1])


@pytest.mark.parametrize("target", ["raw", "event_probability", "business_score"])
@pytest.mark.parametrize("event_label", [0, 1])
def test_lightgbm_optimized(sample, target, event_label):
    x, y = sample
    model = LGBMClassifier(
        n_estimators=10, min_child_samples=5, verbosity=-1, n_jobs=1
    ).fit(x, y)
    result = ModelExplainer(
        model,
        x.iloc[:20],
        target=target,
        event_label=event_label,
        score_aligner=ScoreAligner(),
    ).explain(x.iloc[30:33])
    assert_reconstruction(result)
    if target == "event_probability":
        np.testing.assert_allclose(
            result.predictions, model.predict_proba(x.iloc[30:33])[:, event_label]
        )


def test_reference_schema_reports_and_callable(sample):
    x, _ = sample

    def function(frame):
        return 1 / (1 + np.exp(-frame.income.to_numpy()))

    explainer = ModelExplainer(
        None,
        x,
        prediction_callable=function,
        method="permutation",
        target="event_probability",
        max_background=10,
        max_evals=21,
        feature_dictionary={
            "income": {"display_name": "Annual income", "group": "finance"},
            "debt": {"group": "finance"},
        },
    )
    result = explainer.explain(x.iloc[:2])
    assert_reconstruction(result)
    assert "Annual income" in result.summary()
    assert "high" not in result.summary().lower()
    np.testing.assert_allclose(
        result.aggregate_groups().sum(axis=1), result.contributions.sum(axis=1)
    )
    assert result.global_importance().index[0] == "income"
    assert len(result.top_contributors(top_k=1)["positive"]) <= 1
    assert (
        explainer.reference
        == ModelExplainer(
            None,
            x,
            prediction_callable=function,
            method="permutation",
            target="event_probability",
            max_background=10,
        ).reference
    )
    with pytest.raises(ValueError, match="order"):
        explainer.explain(x[list(reversed(x.columns))].iloc[:1])
    with pytest.raises(ValueError, match="exactly"):
        explainer.explain(x.iloc[:1].assign(extra=1))
    with pytest.raises(TypeError, match="dtypes"):
        explainer.explain(x.iloc[:1].astype("float32"))
    with pytest.raises(ValueError, match="max_rows"):
        explainer.max_rows = 1
        explainer.explain(x.iloc[:2])


def test_native_booster_and_plot(sample, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from lightgbm import Booster

    x, y = sample
    model = LGBMClassifier(
        n_estimators=5, min_child_samples=5, verbosity=-1, n_jobs=1
    ).fit(x, y)
    path = tmp_path / "model.txt"
    model.booster_.save_model(str(path))
    loaded = Booster(model_file=str(path))
    with pytest.raises(ValueError, match="class_labels"):
        ModelExplainer(loaded, x.iloc[:10])
    result = ModelExplainer(loaded, x.iloc[:10], class_labels=[0, 1]).explain(
        x.iloc[20:21]
    )
    assert_reconstruction(result)
    ax = result.waterfall(top_k=1)
    assert "Other features" in [tick.get_text() for tick in ax.get_yticklabels()]


def test_optional_imports():
    code = """
import sys
import brainmodelkit
import brainmodelkit.calibration.pyspark
import brainmodelkit.scoring.pyspark
import brainmodelkit.explainability.pyspark
assert 'shap' not in sys.modules
assert 'pyspark' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
