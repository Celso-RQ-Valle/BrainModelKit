"""Classic Spark integration: SQL contracts, distributed fitting and explanations."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

from brainmodelkit._probability import CapabilityError
from brainmodelkit.calibration.pandas import LogisticCalibrator as PandasCalibrator
from brainmodelkit.calibration.pandas import (
    calibration_diagnostics as pandas_diagnostics,
)
from brainmodelkit.calibration.pyspark import (
    LogisticCalibrator,
    calibration_diagnostics,
)
from brainmodelkit.explainability.pyspark import ModelExplainer, PredictionPipeline
from brainmodelkit.scoring.pandas import ScoreAligner as PandasAligner
from brainmodelkit.scoring.pyspark import ScoreAligner


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    session = (
        SparkSession.builder.master("local[2]")
        .appName("ProbabilityWorkflow")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "2")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_sql_parity_serialization_keys_and_no_python_udf(spark):
    pdf = pd.DataFrame(
        {"client": [1, 1, 2, 2], "month": [1, 2, 1, 2], "score": [0.0, 0.2, 0.8, 1.0]}
    )
    frame = spark.createDataFrame(pdf)
    for space in ("logit", "probability", "raw"):
        local = PandasCalibrator(-0.3, 1.5, input_space=space)
        config = LogisticCalibrator.from_dict(local.to_dict())
        result = config.transform(frame, event_col="event", non_event_col="non_event")
        expected = local.transform(pdf, event_col="event", non_event_col="non_event")
        actual = result.orderBy("client", "month").toPandas()
        np.testing.assert_allclose(actual.event, expected.event)
        np.testing.assert_allclose(actual.non_event, expected.non_event)
        plan = result._jdf.queryExecution().optimizedPlan().toString()
        assert "PythonUDF" not in plan and "pythonUDF" not in plan
    for direction in ("lower_event_probability", "higher_event_probability"):
        align = PandasAligner(
            200,
            800,
            input_probability="non_event",
            higher_score_means=direction,
            decimals=1,
        )
        portable = ScoreAligner.from_dict(align.to_dict())
        actual = portable.transform(frame).orderBy("client", "month").toPandas()
        expected = align.transform(pdf)
        np.testing.assert_allclose(actual.business_score, expected.business_score)
        inverse = (
            portable.inverse_transform(
                spark.createDataFrame(actual), "business_score_unrounded"
            )
            .orderBy("client", "month")
            .toPandas()
        )
        np.testing.assert_allclose(inverse.event_probability, 1 - pdf.score, atol=1e-15)
    halves = spark.createDataFrame(
        [(0.125,), (0.375,), (0.625,), (0.875,)], "score double"
    )
    rounded = (
        ScoreAligner(0, 4, higher_score_means="higher_event_probability", decimals=0)
        .transform(halves)
        .orderBy("score")
        .toPandas()
    )
    assert rounded.business_score.tolist() == [0, 2, 2, 4]
    with pytest.raises(ValueError, match="collision"):
        LogisticCalibrator(0, 1).transform(frame, output_col="score")
    overwritten = (
        LogisticCalibrator(0, 1)
        .transform(frame, output_col="score", event_col="event", overwrite=True)
        .orderBy("client", "month")
        .toPandas()
    )
    np.testing.assert_allclose(overwritten.score, overwritten.event)


def test_invalid_null_policy_and_fitting_parity(spark):
    invalid = spark.createDataFrame(
        [(None,), (float("nan"),), (float("inf"),), (-0.2,), (1.3,)], "score double"
    )
    for config in [LogisticCalibrator(0, 1), ScoreAligner()]:
        with pytest.raises(ValueError):
            config.transform(invalid)
        config.invalid_policy = "null"
        result = config.transform(invalid).toPandas()
        assert result.iloc[:, -1].isna().all()
    x = np.repeat(np.linspace(-2, 2, 15), 2)
    y = np.tile([0, 1], 15)
    p = 1 / (1 + np.exp(-(-0.4 + 1.3 * x)))
    pdf = pd.DataFrame(
        {
            "score": 1 / (1 + np.exp(-x)),
            "target": y,
            "weight": np.where(y == 1, p, 1 - p),
        }
    )
    frame = spark.createDataFrame(pdf)
    local = PandasCalibrator().fit(pdf, weight_col="weight")
    fitted = LogisticCalibrator().fit(frame, weight_col="weight")
    np.testing.assert_allclose(
        [fitted.intercept, fitted.slope], [local.intercept, local.slope], atol=2e-5
    )
    assert PandasCalibrator.from_dict(fitted.to_dict()).mode == "fitted"
    report = calibration_diagnostics(frame, weight_col="weight", n_bins=4)
    expected = pandas_diagnostics(pdf, weight_col="weight", n_bins=4)
    assert report["brier_score"] == pytest.approx(expected["brier_score"])
    assert report["log_loss"] == pytest.approx(expected["log_loss"])
    np.testing.assert_allclose(
        report["reliability"].toPandas().event_rate, expected["reliability"].event_rate
    )
    from pyspark.sql import functions as F

    with pytest.raises(ValueError, match="Constant"):
        LogisticCalibrator().fit(frame.withColumn("score", F.lit(0.5)))


@pytest.fixture(scope="module")
def logistic(spark):
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import VectorAssembler

    data = spark.createDataFrame(
        [
            (i // 2, i % 2, float(i % 7), float(i % 5), int(i % 3 == 0))
            for i in range(40)
        ],
        "client long, month long, income double, debt double, target int",
    )
    model = Pipeline(
        stages=[
            VectorAssembler(inputCols=["income", "debt"], outputCol="features"),
            LogisticRegression(labelCol="target", maxIter=50),
        ]
    ).fit(data)
    return data, model


def test_logistic_distributed_and_local_complete_explanations(spark, logistic):
    data, model = logistic
    features = ["income", "debt"]
    background = data.select(*features)
    rows = data.drop("target").where("client < 2")
    cal = LogisticCalibrator(-0.2, 1.2, output_probability="non_event")
    align = ScoreAligner(300, 850, decimals=0)
    predicted = (
        PredictionPipeline(model, features, calibrator=cal, score_aligner=align)
        .transform(rows)
        .orderBy("client", "month")
        .toPandas()
    )
    options = dict(
        key_cols=["client", "month"],
        target="business_score",
        method="permutation",
        max_background=5,
        max_evals=15,
        calibrator=cal,
        score_aligner=align,
        max_rows=10,
    )
    distributed = ModelExplainer(model, background, features, **options).explain(rows)
    result = distributed.to_frame().orderBy("client", "month").toPandas()
    np.testing.assert_allclose(
        result.prediction, predicted.business_score_unrounded, atol=1e-6
    )
    assert result.reconstruction_ok.all()
    local = ModelExplainer(
        model, background, features, execution="local", **options
    ).explain(rows.orderBy("client", "month"))
    np.testing.assert_allclose(result.prediction, local.predictions, atol=1e-8)
    assert distributed.global_importance().count() == 2
    assert distributed.top_contributors(1).count() <= 8
    assert (
        distributed.aggregate_groups({"income": "finance", "debt": "finance"}).count()
        == 4
    )
    with pytest.raises(ValueError, match="max_rows"):
        distributed.local(1)
    raw = ModelExplainer(
        model,
        background,
        features,
        key_cols=["client", "month"],
        target="raw",
        execution="local",
    ).explain(rows)
    assert np.max(np.abs(raw.residuals)) < 1e-6
    with pytest.raises(CapabilityError, match="Unsupported Spark"):
        ModelExplainer(object(), background, features, key_cols=["client", "month"])


def test_sparklightgbm_native_adapter(spark):
    """Exercise the actual wrapper/native representation without Spark training."""
    from lightgbm import LGBMClassifier
    from pyspark.ml import PipelineModel
    from pyspark.ml.feature import VectorAssembler

    from brainmodelkit.training._spark_lightgbm import SparkLightGBMModel

    pdf = pd.DataFrame(
        {"income": np.arange(40, dtype=float), "debt": np.arange(40, dtype=float) % 7}
    )
    model = LGBMClassifier(
        n_estimators=5, min_child_samples=2, verbosity=-1, n_jobs=1
    ).fit(pdf, (pdf.income % 3 == 0).astype(int))
    native = SparkLightGBMModel(nativeModel=model.booster_.model_to_string())
    pipeline = PipelineModel(
        [VectorAssembler(inputCols=list(pdf), outputCol="features"), native]
    )
    background = spark.createDataFrame(pdf)
    rows = spark.createDataFrame(pdf.iloc[:3].assign(client=[1, 1, 2], month=[1, 2, 1]))
    explainer = ModelExplainer(
        pipeline,
        background,
        list(pdf),
        key_cols=["client", "month"],
        target="event_probability",
        max_background=10,
    )
    result = explainer.explain(rows).to_frame().orderBy("client", "month").toPandas()
    np.testing.assert_allclose(
        result.prediction, model.predict_proba(pdf.iloc[:3])[:, 1]
    )
    assert result.reconstruction_ok.all()
