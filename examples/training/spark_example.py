"""Run with a working local Spark/Java installation and .[pyspark]."""

from pyspark.sql import SparkSession

from brainmodelkit.pyspark import simulate_credit_data
from brainmodelkit.training.pyspark import train_model

spark = SparkSession.builder.master("local[2]").appName("TrainingExample").getOrCreate()
try:
    data, features = simulate_credit_data(spark, row_count=1000, feature_count=6)
    train, oot = data.randomSplit([0.75, 0.25], seed=42)
    result = train_model(
        run_name="random_forest_demo",
        target_col="default_flag",
        feature_cols=features,
        train_df=train,
        oot_df=oot,
        model="random_forest",
        model_params={"numTrees": 20, "maxDepth": 5, "seed": 42},
        df_scoring=oot.drop("default_flag"),
    )
    print(result.metrics)
    print(result.output_dir)
    result.scoring_predictions.show(5)
finally:
    spark.stop()
