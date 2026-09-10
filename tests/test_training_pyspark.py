"""Integration coverage for the fitted raw-feature Spark pipeline."""

import os
import sys

import pytest
from pyspark.sql import SparkSession

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
def test_training(spark, tmp_path, model):
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
