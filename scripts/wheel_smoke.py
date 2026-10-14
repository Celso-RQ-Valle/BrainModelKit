"""Real model lifecycle checks; run with -I against an installed wheel."""

import sys
import sysconfig
from pathlib import Path

import brainmodelkit
from brainmodelkit.persistence import load_model


def pandas_round_trip():
    import numpy as np
    import pandas as pd

    from brainmodelkit.metrics.pandas import auc_gini
    from brainmodelkit.training.pandas import train_model

    train = pd.DataFrame(
        {"x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], "y": [0, 0, 0, 1, 1, 1]}
    )
    oot = pd.DataFrame({"x": [-2.5, -0.5, 0.5, 2.5], "y": [0, 0, 1, 1]})
    result = train_model("wheel", "y", ["x"], train, oot, output_dir="reports")
    assert result.metrics["oot_auc"] == 1.0
    assert auc_gini(df=result.oot_predictions, target_column="y")["auc"].iloc[0] == 1.0
    loaded = load_model(result.model_uri, "pickle")
    np.testing.assert_allclose(
        loaded.predict_proba(oot[["x"]])[:, 1], result.oot_predictions["score"]
    )
    assert (result.output_dir / "metrics.json").is_file()


def spark_round_trip():
    import numpy as np
    from pyspark.ml import PipelineModel
    from pyspark.sql import SparkSession

    from brainmodelkit.metrics.pyspark import auc_gini
    from brainmodelkit.training.pyspark import train_model

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("WheelLifecycle")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    try:
        train = spark.createDataFrame(
            [(-3.0, 0), (-2.0, 0), (-1.0, 0), (1.0, 1), (2.0, 1), (3.0, 1)],
            "x double, y int",
        )
        oot = spark.createDataFrame(
            [(-2.5, 0), (-0.5, 0), (0.5, 1), (2.5, 1)], "x double, y int"
        )
        result = train_model(
            "wheel", "y", ["x"], train, oot, output_dir="reports", n_tiles=4
        )
        assert result.metrics["oot_auc"] == 1.0
        assert (
            auc_gini(df=result.oot_predictions, target_column="y", n_tiles=4)
            .first()
            .auc
            == 1.0
        )
        loaded = load_model(result.model_uri, "spark", model_class=PipelineModel)
        before = result.model.transform(oot.drop("y")).orderBy("x").collect()
        after = loaded.transform(oot.drop("y")).orderBy("x").collect()
        assert len(after) == 4
        np.testing.assert_allclose(
            [row.probability for row in before], [row.probability for row in after]
        )
        assert [row.prediction for row in before] == [row.prediction for row in after]
        assert (result.output_dir / "metrics.json").is_file()
    finally:
        spark.stop()


if __name__ == "__main__":
    installed = Path(brainmodelkit.__file__).resolve()
    assert installed.is_relative_to(Path(sysconfig.get_paths()["purelib"]).resolve()), (
        installed
    )
    {"pandas": pandas_round_trip, "spark": spark_round_trip}[sys.argv[1]]()
    print(f"Wheel {sys.argv[1]} lifecycle passed: {installed}")
