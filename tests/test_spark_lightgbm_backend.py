"""Backend factory contracts and optional real SparkLightGBM round trips."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from brainmodelkit.training._spark_estimators import make_estimator, validate_backend


def test_default_uses_public_snake_case_api(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "sparklightgbm", SimpleNamespace(__version__="0.9.2")
    )
    estimator = make_estimator("lightgbm", {"num_iterations": 3}, "y", 17)
    options = estimator.getOrDefault(estimator.options)
    assert options["num_iterations"] == 3
    assert options["label_col"] == "y"
    assert options["seed"] == 17
    assert options["probability_col"] == "probability"
    assert options["leaf_prediction_col"] is None
    assert type(make_estimator("lgbm", {}, "y")) is type(estimator)
    with pytest.raises(ValueError, match="snake_case"):
        make_estimator("lightgbm", {"numIterations": 3}, "y")


def test_no_silent_backend_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "sparklightgbm", None)
    with pytest.raises(ImportError, match=r"brainmodelkit\[sparklightgbm\]"):
        make_estimator("lightgbm", {}, "y")


@pytest.mark.parametrize("version", ["0.9.0", "0.10.0"])
def test_unsupported_bridge_version(monkeypatch, version):
    monkeypatch.setitem(
        sys.modules, "sparklightgbm", SimpleNamespace(__version__=version)
    )
    with pytest.raises(ImportError, match="Unsupported SparkLightGBM version"):
        make_estimator("lightgbm", {}, "y")


def test_custom_estimator_is_copied_and_parameters_preserved():
    original = Mock()
    copied = original.copy.return_value
    copied.hasParam.return_value = True
    copied.isSet.return_value = True
    result = make_estimator(original, {"regParam": 0.1}, "y")
    assert result is copied
    copied.setParams.assert_called_once_with(regParam=0.1)
    original.setParams.assert_not_called()


def test_explicit_synapse_backend(monkeypatch):
    constructor = Mock()
    constructor.return_value.hasParam.return_value = True
    monkeypatch.setitem(
        sys.modules,
        "synapse.ml.lightgbm",
        SimpleNamespace(LightGBMClassifier=constructor),
    )
    make_estimator("lightgbm", {"numIterations": 3}, "y", lgbm_backend="synapseml")
    constructor.assert_called_once_with(numIterations=3)


def test_backend_not_a_model_parameter():
    with pytest.raises(ValueError, match="model_params"):
        validate_backend("default", {"lgbm_backend": "synapseml"})
    with pytest.raises(ValueError, match="default or synapseml"):
        validate_backend("unknown")


@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("sparklightgbm")
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    session = (
        SparkSession.builder.master("local[2]").appName("BMKLightGBM").getOrCreate()
    )
    yield session
    session.stop()


def test_real_bridge_pipeline_probability_and_cv(spark):
    from pyspark.ml.linalg import VectorUDT

    from brainmodelkit.model_selection.pyspark import cross_validate
    from brainmodelkit.training.pyspark import train_model

    data = spark.createDataFrame(
        [(float(i), int(i > 19)) for i in range(40)], "x double, y int"
    )
    params = {
        "num_iterations": 3,
        "min_data_in_leaf": 1,
        "num_threads": 1,
        "verbosity": -1,
    }
    result = train_model(
        "bridge",
        "y",
        ["x"],
        data,
        data,
        model="lightgbm",
        model_params=params,
        save_model_to="none",
        n_tiles=4,
    )
    assert isinstance(result.oot_predictions.schema["probability"].dataType, VectorUDT)
    rows = result.model.transform(data).select("probability", "prediction").collect()
    assert all(sum(row.probability) == pytest.approx(1) for row in rows)
    assert all(0 <= row.probability[1] <= 1 for row in rows)
    assert result.metrics["oot_ks"] > 0.85
    # Exercise exactly the metadata serializer used by Spark's Python-stage
    # writer even on Windows hosts without Hadoop's local file writer.
    import json

    from pyspark.ml import PipelineModel
    from pyspark.ml.util import DefaultParamsReader, DefaultParamsWriter

    from brainmodelkit.training._spark_lightgbm import SparkLightGBMModel

    stage = result.model.stages[-1]
    metadata = json.loads(
        DefaultParamsWriter._get_metadata_to_save(stage, spark.sparkContext)
    )
    restored_stage = SparkLightGBMModel()
    DefaultParamsReader.getAndSetParams(restored_stage, metadata)
    restored = PipelineModel(stages=[result.model.stages[0], restored_stage])
    assert (
        restored.transform(data).select("probability", "prediction").collect() == rows
    )
    cv = cross_validate(
        data,
        "y",
        ["x"],
        "lightgbm",
        model_params=params,
        cv=2,
        metrics=("ks",),
        n_tiles=4,
    )
    assert len(cv.fold_metrics) == 2
    assert cv.summary[0]["mean"] > 0.8


@pytest.mark.skipif(
    os.name == "nt" and not os.environ.get("HADOOP_HOME"),
    reason="Spark disk writes on Windows require Hadoop/winutils",
)
@pytest.mark.parametrize("format", ["spark", "mlflow"])
def test_real_bridge_pipeline_persistence(spark, tmp_path, format):
    from pyspark.ml import PipelineModel

    from brainmodelkit.training._spark_estimators import make_pipeline

    data = spark.createDataFrame(
        [(float(i), int(i > 9)) for i in range(20)], "x double, y int"
    )
    fitted = make_pipeline(
        ["x"], "y", "lightgbm", {"num_iterations": 2, "num_threads": 1}
    ).fit(data)
    path = str(tmp_path / "pipeline")
    if format == "spark":
        fitted.write().save(path)
        restored = PipelineModel.load(path)
    else:
        mlflow_spark = pytest.importorskip("mlflow.spark")
        mlflow_spark.save_model(fitted, path)
        restored = mlflow_spark.load_model(path)
    assert (
        fitted.transform(data).select("probability").collect()
        == restored.transform(data).select("probability").collect()
    )


def test_mlflow_accepts_python_pipeline_stage():
    mlflow_spark = pytest.importorskip("mlflow.spark")
    from pyspark.ml import PipelineModel

    from brainmodelkit.training._spark_lightgbm import SparkLightGBMModel

    mlflow_spark._validate_model(PipelineModel(stages=[SparkLightGBMModel()]))


@pytest.mark.parametrize("backend", ["default", "synapseml"])
@pytest.mark.parametrize("no_feasible", [False, True])
def test_optimizer_propagates_backend_and_refits_selected_trial(
    monkeypatch, backend, no_feasible
):
    from unittest.mock import MagicMock

    import brainmodelkit.optimization.pyspark as module

    pytest.importorskip("optuna")
    monkeypatch.setattr(module, "_validate", lambda *args: None)
    cv = Mock(return_value=SimpleNamespace(summary=[{"metric": "ks", "mean": 0.3}]))
    pipeline = MagicMock()
    final = Mock(return_value=SimpleNamespace(model="selected-model"))
    monkeypatch.setattr(module, "cross_validate", cv)
    monkeypatch.setattr(module, "_pipeline", pipeline)
    monkeypatch.setattr(module, "train_model", final)
    monkeypatch.setattr("pyspark.ml.functions.vector_to_array", MagicMock())
    metrics = iter([0.8, 1.0, 0.4, 0.43])
    monkeypatch.setattr(module, "_metrics", lambda *args: {"ks": next(metrics)})

    def optimize():
        return module.optimize(
            object(),
            object(),
            "y",
            ["x"],
            "lightgbm",
            {"num_leaves": lambda t: t.number + 2},
            objective_metric="oot_ks",
            target_metric="train_oot_ks_abs_shift",
            target_limit=0.01 if no_feasible else 0.05,
            target_operator="<=",
            lgbm_backend=backend,
            metrics=("ks",),
            n_trials=2,
            random_state=7,
        )

    if no_feasible:
        with pytest.warns(RuntimeWarning, match="No feasible trial"):
            result = optimize()
    else:
        result = optimize()
    assert result.selected_trial_number == 1
    assert result.best_value == 0.4
    assert result.best_params == {"num_leaves": 3}
    assert result.best_model == "selected-model"
    assert result.constraint_satisfied is (not no_feasible)
    assert all(call.kwargs["lgbm_backend"] == backend for call in cv.call_args_list)
    assert all(call.args[-1] == backend for call in pipeline.call_args_list)
    assert final.call_args.kwargs["lgbm_backend"] == backend
    assert final.call_args.kwargs["random_state"] == 7
    assert final.call_args.kwargs["model_params"] == result.best_params
