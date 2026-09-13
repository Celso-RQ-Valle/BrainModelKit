"""Native Spark filters and model selection; input rows remain distributed."""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from brainmodelkit.feature_selection.pyspark import (
    completeness,
    correlation_filter,
    feature_importance_selection,
    information_value,
    variance_filter,
)

spark = SparkSession.builder.appName("NativeFeatureSelection").getOrCreate()
try:
    df = spark.range(300).select(
        (F.col("id") % 2).cast("int").alias("target"),
        ((F.col("id") % 2) + F.rand(42)).alias("signal"),
        F.rand(7).alias("noise"),
        F.lit(1.0).alias("constant"),
    )
    features = ["signal", "noise", "constant"]
    features = completeness(df, features).selected_features
    features = variance_filter(df, features).selected_features
    features = correlation_filter(df, features).selected_features
    iv = information_value(df, "target", features, min_iv=0.02)
    iv.bin_details.show(truncate=False)
    if not iv.selected_features:
        raise ValueError("No features passed the screening thresholds")
    result = feature_importance_selection(
        df,
        "target",
        iv.selected_features,
        model="random_forest",
        model_params={"numTrees": 10},
        top_k=1,
    )
    print(result.selected_features)
finally:
    spark.stop()
