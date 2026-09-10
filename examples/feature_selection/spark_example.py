"""Execute Spark RFE with a generated binary dataset."""

from pyspark.sql import SparkSession

from brainmodelkit.feature_selection.pyspark import rfe
from brainmodelkit.pyspark import simulate_credit_data

spark = SparkSession.builder.master("local[2]").appName("SparkRFE").getOrCreate()
try:
    data, features = simulate_credit_data(spark, row_count=1000, feature_count=6)
    train, oot = data.randomSplit([0.75, 0.25], seed=42)
    selection = rfe(
        "spark_random_forest_rfe",
        "default_flag",
        train,
        oot,
        feature_cols=features,
        model="random_forest",
        model_params={"numTrees": 20, "maxDepth": 5, "seed": 42},
        step=2,
        n_feat_final=3,
    )
    print(selection.selected_features)
    print(selection.training_result.metrics)
    print(selection.output_dir)
finally:
    spark.stop()
