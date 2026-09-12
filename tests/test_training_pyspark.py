"""Integration coverage for the fitted raw-feature Spark pipeline."""

import os
import sys

import pytest
from pyspark.sql import SparkSession

from brainmodelkit.persistence import load_model
from brainmodelkit.training.pyspark import train_model


@pytest.fixture(scope="module")
def spark():
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    session = (
        SparkSession.builder.master("local[1]").appName("TrainingTests").getOrCreate()
    )
    yield session
    session.stop()


@pytest.mark.parametrize(
    "model", ["logistic_regression", "random_forest", "gradient_boosting"]
)
def test_training(spark, tmp_path, model, monkeypatch):
    def unexpected_save(*args, **kwargs):
        pytest.fail("Training without save_path must not persist the model")

    monkeypatch.setattr(
        "brainmodelkit.training.pyspark._save_spark_model", unexpected_save
    )
    data = spark.createDataFrame(
        [(-3.0, 0), (-2.0, 0), (-1.0, 0), (1.0, 1), (2.0, 1), (3.0, 1)],
        "x double, target int",
    )
    result = train_model(
        "spark",
        "target",
        ["x"],
        data,
        data,
        model=model,
        df_scoring=data.drop("target"),
        n_tiles=6,
        output_dir=tmp_path,
    )
    assert result.metrics["oot_auc"] == pytest.approx(1)
    assert result.metrics["oot_ks"] == pytest.approx(1)
    assert result.scoring_predictions.count() == 6
    assert result.model.transform(data.drop("target")).count() == 6
    assert len(result.feature_importance) == 1


def test_training_persistence(spark, tmp_path):
    from pyspark.ml import PipelineModel

    data = spark.createDataFrame(
        [(-2.0, 0), (-1.0, 0), (1.0, 1), (2.0, 1)], "x double, target int"
    )
    path = tmp_path / "pipeline"
    result = train_model(
        "saved",
        "target",
        ["x"],
        data,
        data,
        output_dir=tmp_path / "reports",
        save_path=path,
        n_tiles=4,
    )
    loaded = load_model(path, "spark", model_class=PipelineModel)
    assert loaded.transform(data).select("prediction").collect() == (
        result.model.transform(data).select("prediction").collect()
    )
    assert (tmp_path / "pipeline.metadata.json").exists()
    with pytest.raises(ValueError, match="Available:"):
        train_model("bad", "target", ["x"], data, data, save_format="pickle")


@pytest.mark.parametrize(
    "model", ["logistic_regression", "random_forest", "gradient_boosting"]
)
def test_rfe(spark, tmp_path, model):
    from brainmodelkit.feature_selection.pyspark import rfe

    data = spark.createDataFrame(
        [
            (-3.0, 0.0, 0),
            (-2.0, 0.0, 0),
            (-1.0, 0.0, 0),
            (1.0, 0.0, 1),
            (2.0, 0.0, 1),
            (3.0, 0.0, 1),
        ],
        "feat_signal double, feat_constant double, target int",
    )
    result = rfe(
        "spark_rfe",
        "target",
        data,
        data,
        model=model,
        n_feat_final=1,
        step=2,
        n_tiles=6,
        output_dir=tmp_path,
        df_scoring=data.drop("target"),
    )
    assert result.selected_features == ["feat_signal"]
    assert [h["n_features"] for h in result.history] == [2, 1]
    assert result.training_result.scoring_predictions.count() == 6
    assert result.training_result.model.stages[0].getInputCols() == ["feat_signal"]
