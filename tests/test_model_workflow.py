"""Real estimator and artifact parity across independently optional integrations."""

import json
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from brainmodelkit.pandas import (
    LogisticCalibrator,
    ModelExplainer,
    PredictionPipeline,
    ScoreAligner,
)
from brainmodelkit.persistence import ArtifactSpec, load_workflow, save_workflow
from brainmodelkit.persistence.python import _save_python_model


@pytest.fixture
def data():
    rng = np.random.default_rng(4)
    x = pd.DataFrame(rng.normal(size=(60, 2)), columns=["a", "b"])
    return x, (x.a + x.b > 0).astype(int)


@pytest.mark.parametrize("format", ["pickle", "joblib", "cloudpickle", "skops"])
def test_bundle(data, tmp_path, format, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    x, y = data
    model = LogisticRegression().fit(x, y)
    workflow = PredictionPipeline(
        model,
        event_label=0,
        calibrator=LogisticCalibrator(0.3, 1.2, event_label=0),
        score_aligner=ScoreAligner(decimals=1),
    )
    path = tmp_path / "bundle"
    save_workflow(path, workflow, format=format, schema=x.dtypes.astype(str).to_dict())
    loaded = load_workflow(path, format=format)
    np.testing.assert_allclose(loaded.predict(x), model.predict_proba(x)[:, 0])
    result = ModelExplainer(
        loaded, x.head(5), target="business_score", method="permutation", max_evals=9
    ).explain(x.head(2).assign(target=1))
    np.testing.assert_allclose(
        result.predictions, workflow.predict(x.head(2), "business_score")
    )
    manifest = path / "workflow.json"
    meta = json.loads(manifest.read_text())
    meta["class_labels"] = [1, 0]
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Conflicting"):
        load_workflow(path, format=format)


@pytest.mark.parametrize(
    "cls", [DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier]
)
def test_optimized_trees(data, cls):
    x, y = data
    model = cls(random_state=4).fit(x, y)
    for event in (0, 1):
        result = ModelExplainer(
            model, x.head(12), event_label=event, target="event_probability"
        ).explain(x.tail(3))
        np.testing.assert_allclose(
            result.predictions, model.predict_proba(x.tail(3))[:, event]
        )
        assert result.to_frame().reconstruction_ok.all()


@pytest.mark.parametrize("zipmap", [True, False])
def test_onnx_tensor_and_map(data, zipmap, tmp_path):
    import skl2onnx

    x, y = data
    model = LogisticRegression().fit(x, y)
    path = tmp_path / "classifier.onnx"
    path.write_bytes(
        skl2onnx.to_onnx(
            model,
            x.to_numpy(dtype=np.float32)[:1],
            options={id(model): {"zipmap": zipmap}},
            target_opset=17,
        ).SerializeToString()
    )
    workflow = PredictionPipeline(
        ArtifactSpec(path=path, format="onnx"),
        list(x),
        class_labels=[0, 1],
        event_label=0,
    )
    np.testing.assert_allclose(
        workflow.predict(x), model.predict_proba(x)[:, 0], atol=1e-6
    )
    if zipmap:
        wrong_classes = PredictionPipeline(
            workflow.model, list(x), class_labels=[0, 7], event_label=0
        )
        with pytest.raises(ValueError, match="map keys"):
            wrong_classes.predict(x)
    result = ModelExplainer(
        workflow,
        x.head(6),
        target="event_probability",
        method="permutation",
        max_evals=9,
    ).explain(x.tail(2))
    assert result.to_frame().reconstruction_ok.all()


def test_xgb_native(data, tmp_path):
    import xgboost as xgb

    x, y = data
    model = xgb.XGBClassifier(n_estimators=4, n_jobs=1).fit(x, y)
    path = tmp_path / "tree.ubj"
    _save_python_model(model, path, "native")
    spec = ArtifactSpec(path=path, format="native", model_class=xgb.Booster)
    workflow = PredictionPipeline(spec, list(x), class_labels=[0, 1], event_label=0)
    np.testing.assert_allclose(
        workflow.predict(x), PredictionPipeline(model, event_label=0).predict(x)
    )
    np.testing.assert_allclose(
        workflow.predict(x, "raw"), -model.predict(x, output_margin=True)
    )


def test_mixed_preprocessing(data):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    x, y = data
    x = x.assign(group=np.where(x.b > 0, "first", "second")).drop(columns="b")
    model = make_pipeline(
        ColumnTransformer(
            [("group", OneHotEncoder(handle_unknown="ignore"), ["group"])],
            remainder="passthrough",
        ),
        LogisticRegression(),
    ).fit(x, y)
    rows = x.tail(3).assign(group=["new", None, "first"])
    result = ModelExplainer(
        model, x.head(10), target="event_probability", method="permutation", max_evals=9
    ).explain(rows.assign(unrelated=1))
    np.testing.assert_allclose(result.predictions, model.predict_proba(rows)[:, 1])


def test_mlflow_native_flavors(data, tmp_path):
    import mlflow.sklearn
    import mlflow.xgboost
    from xgboost import XGBClassifier

    x, y = data
    for model, module, flavor in [
        (LogisticRegression(), mlflow.sklearn, "sklearn"),
        (XGBClassifier(n_estimators=3, n_jobs=1), mlflow.xgboost, "xgboost"),
    ]:
        model.fit(x, y)
        path = tmp_path / flavor
        module.save_model(model, str(path), pip_requirements=[])
        loaded = PredictionPipeline(
            ArtifactSpec(path=path, format="mlflow", flavor=flavor), list(x)
        )
        np.testing.assert_allclose(loaded.predict(x), model.predict_proba(x)[:, 1])


def test_reference_bundle_and_adapter(data, tmp_path):
    from brainmodelkit.prediction.adapters import register_adapter

    x, y = data
    model = LogisticRegression().fit(x, y)
    path = tmp_path / "external.pkl"
    _save_python_model(model, path, "pickle")
    ref = ArtifactSpec(path=path, format="pickle")
    pipeline = PredictionPipeline(ref)
    save_workflow(tmp_path / "external", pipeline, reference=ref)
    with pytest.raises(ValueError, match="external ArtifactSpec"):
        load_workflow(tmp_path / "external", format="pickle")
    copied = load_workflow(tmp_path / "external", format="pickle", reference=ref)
    np.testing.assert_allclose(copied.predict(x), pipeline.predict(x))

    class Wrapped:
        classes_ = (0, 1)

    class Adapter:
        def __init__(self, wrapped, index):
            self.index = index

        def probability(self, frame):
            return model.predict_proba(frame)[:, self.index]

    register_adapter(Wrapped, Adapter)
    custom = PredictionPipeline(Wrapped(), list(x), event_label=0)
    np.testing.assert_allclose(custom.predict(x), model.predict_proba(x)[:, 0])


def test_catboost_native(data, tmp_path):
    from catboost import CatBoostClassifier

    x, y = data
    model = CatBoostClassifier(
        iterations=5, thread_count=1, verbose=False, allow_writing_files=False
    ).fit(x, y)
    path = tmp_path / "model.cbm"
    _save_python_model(model, path, "native")
    flow = PredictionPipeline(
        ArtifactSpec(path=path, format="native", model_class=CatBoostClassifier),
        list(x),
        event_label=0,
    )
    np.testing.assert_allclose(flow.predict(x), model.predict_proba(x)[:, 0])
    np.testing.assert_allclose(
        flow.predict(x, "raw"), -model.predict(x, prediction_type="RawFormulaVal")
    )


def test_pyfunc_requires_declared_probability(data, tmp_path):
    import mlflow.pyfunc

    x, _ = data

    class Probabilities(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return pd.DataFrame({"credit": 1 / (1 + np.exp(-model_input.a))})

    path = tmp_path / "pyfunc"
    mlflow.pyfunc.save_model(
        str(path), python_model=Probabilities(), pip_requirements=[]
    )
    spec = ArtifactSpec(path=path, format="mlflow", flavor="pyfunc")
    with pytest.raises(ValueError):
        PredictionPipeline(spec, list(x))
    workflow = PredictionPipeline(spec, list(x), probability_output="credit")
    np.testing.assert_allclose(workflow.predict(x), 1 / (1 + np.exp(-x.a)))


def test_embedded_background_integrity(data, tmp_path):
    x, y = data
    model = LogisticRegression().fit(x, y)
    pipeline = PredictionPipeline(model)
    folder = tmp_path / "bundle"
    save_workflow(folder, pipeline, background=x.head(4).assign(private_id=1))
    loaded = load_workflow(folder, format="pickle")
    pd.testing.assert_frame_equal(loaded.background, x.head(4))
    (folder / "background.parquet").write_bytes(b"other")
    with pytest.raises(ValueError, match="integrity"):
        load_workflow(folder, format="pickle")


def test_raw_only_callable_calibration(data):
    x, _ = data
    pipeline = PredictionPipeline(
        None,
        list(x),
        raw_callable=lambda frame: frame.a.to_numpy(),
        calibrator=LogisticCalibrator(0, 1, input_space="raw", input_probability=None),
    )
    actual = pipeline.predict(x, "calibrated_event_probability")
    np.testing.assert_allclose(actual, 1 / (1 + np.exp(-x.a)))


@pytest.mark.parametrize("framework", ["catboost", "lightgbm", "onnx"])
def test_other_mlflow_native_flavors(data, tmp_path, framework):
    import importlib

    x, y = data
    module = importlib.import_module("mlflow." + framework)
    model = None
    options = {}
    if framework == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=3, thread_count=1, verbose=False, allow_writing_files=False
        ).fit(x, y)
    elif framework == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=4, min_child_samples=2, verbosity=-1, n_jobs=1
        ).fit(x, y)
    else:
        from skl2onnx import to_onnx

        model = LogisticRegression().fit(x, y)
        options["class_labels"] = [0, 1]
    saved = (
        to_onnx(model, x.to_numpy(np.float32)[:1], target_opset=17)
        if framework == "onnx"
        else model
    )
    path = tmp_path / framework
    module.save_model(saved, str(path), pip_requirements=[])
    loaded = PredictionPipeline(
        ArtifactSpec(path=path, format="mlflow", flavor=framework), list(x), **options
    )
    np.testing.assert_allclose(
        loaded.predict(x), model.predict_proba(x)[:, 1], atol=1e-6
    )


def test_uri_and_schema_contract(data, tmp_path):
    from brainmodelkit.persistence import load_model, resolve_model

    x, y = data
    model = LogisticRegression().fit(x, y)
    with pytest.raises(ValueError, match="MLflow"):
        load_model("models:/credit/1", format="joblib")
    with pytest.raises(ValueError, match="mounted"):
        load_model("s3://bucket/credit.pkl", format="pickle")
    with pytest.raises(TypeError, match="ArtifactSpec"):
        resolve_model(str(tmp_path / "model.pkl"))
    workflow = PredictionPipeline(model, event_label=0, score_aligner=ScoreAligner())
    copied = PredictionPipeline(workflow)
    np.testing.assert_allclose(
        workflow.predict(x, "business_score"), copied.predict(x, "business_score")
    )
    with pytest.raises(ValueError, match="Missing"):
        copied.predict(x.drop(columns="a"))
    with pytest.raises(ValueError, match="Duplicate"):
        copied.predict(pd.concat([x, x], axis=1))


def test_conflicting_artifact_schema(data, tmp_path):
    x, y = data
    path = tmp_path / "model.joblib"
    _save_python_model(LogisticRegression().fit(x, y), path, "joblib")
    sidecar = tmp_path / "model.joblib.metadata.json"
    record = json.loads(sidecar.read_text())
    record["schema"] = {"a": "int64", "b": "float64"}
    sidecar.write_text(json.dumps(record))
    (tmp_path / "model_info.json").write_text(
        json.dumps({"schema": x.dtypes.astype(str).to_dict()})
    )
    with pytest.raises(ValueError, match="Conflicting artifact schema"):
        PredictionPipeline(ArtifactSpec(path=path, format="joblib"))


def test_callable_needs_declared_features(data):
    x, y = data
    with pytest.raises(ValueError, match="feature_cols"):
        ModelExplainer(
            None, x.assign(target=y), prediction_callable=lambda df: df.target
        )


def test_predict_without_shap(data, monkeypatch):
    x, y = data
    model = LogisticRegression().fit(x, y)
    monkeypatch.setitem(sys.modules, "shap", None)
    workflow = PredictionPipeline(model, score_aligner=ScoreAligner())
    assert len(workflow.transform(x)) == len(x)
    with pytest.raises(ImportError, match="explainability"):
        ModelExplainer(workflow, x.head(4)).explain(x.head(1))


def test_mask_preserves_distinct_missing_and_large_integers():
    frame = pd.DataFrame(
        {
            "category": pd.Series(["x", None, float("nan")], dtype=object),
            "amount": np.array([2**54 + 1, 2**54, 2**54 + 3], dtype=np.int64),
        }
    )

    def event(batch):
        return (
            0.1
            + 0.2 * batch.category.map(lambda value: value is None)
            + 0.3 * (batch.amount % 2)
        )

    flow = PredictionPipeline(None, list(frame), prediction_callable=event)
    result = ModelExplainer(
        flow, frame, method="permutation", target="event_probability", max_evals=9
    ).explain(frame)
    np.testing.assert_allclose(result.predictions, event(frame))
    assert result.to_frame().reconstruction_ok.all()
