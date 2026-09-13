"""Small native Spark CV example."""

from pyspark.sql import SparkSession

from brainmodelkit.model_selection.pyspark import cross_validate

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame([(float(i), i % 2) for i in range(20)], ["x", "target"])
result = cross_validate(df, "target", ["x"], "logistic_regression", cv=2, n_tiles=2)
print(result.fold_metrics)
print(result.summary)
