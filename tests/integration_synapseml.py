"""Run explicitly on classic Spark with the matching SynapseML JVM artifacts."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from synapse.ml.lightgbm import LightGBMClassificationModel

from brainmodelkit.pyspark import ModelExplainer, PredictionPipeline


def test_synapse_native_export():
    spark = SparkSession.builder.appName("SynapseExportParity").getOrCreate()
    try:
        frame = pd.DataFrame(
            {"x": np.arange(24, dtype=float), "z": np.arange(24) % 3.0}
        )
        model = LGBMClassifier(
            n_estimators=5, min_child_samples=2, verbosity=-1, n_jobs=1
        ).fit(frame, (frame.x % 4 == 0).astype(int))
        native = LightGBMClassificationModel.loadNativeModelFromString(
            model.booster_.model_to_string()
        )
        assembler = VectorAssembler(
            inputCols=list(frame), outputCol=native.getFeaturesCol()
        )
        workflow = PredictionPipeline(PipelineModel([assembler, native]), list(frame))
        data = spark.createDataFrame(frame.assign(key=frame.index))
        rows = workflow.transform(data).where("key < 3")
        explained = ModelExplainer(
            workflow,
            data,
            key_cols=["key"],
            target="event_probability",
            execution="local",
            max_rows=3,
            max_background=6,
        ).explain(rows)
        expected = rows.select("key", "event_probability").toPandas().set_index("key")
        np.testing.assert_allclose(
            explained.predictions, expected.loc[explained.keys.key, "event_probability"]
        )
    finally:
        spark.stop()
