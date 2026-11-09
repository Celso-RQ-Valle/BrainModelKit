"""Simulate provider returns and validate coverage and a monthly score shift."""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from brainmodelkit.pyspark import validate_external_dataframe


def main():
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("ExternalProviderValidation")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    customers = spark.range(100).select(F.col("id").alias("customer_id"))
    months = spark.createDataFrame(
        [("2026-01-01",), ("2026-02-01",), ("2026-03-01",)], ["reference_date"]
    ).withColumn("reference_date", F.to_date("reference_date"))
    df_sent = customers.crossJoin(months)
    df_received = df_sent.filter("customer_id >= 5").withColumn(
        "bureau_score",
        F.when(F.col("customer_id") % 17 == 0, F.lit(None).cast("double")).otherwise(
            150.0
            + (F.col("customer_id") % 30) * 10
            + (F.month("reference_date") - 1) * 200
        ),
    )
    unexpected = df_received.filter("customer_id = 99").withColumn(
        "customer_id", F.lit(999).cast("long")
    )
    duplicates = df_received.filter("customer_id = 20")
    df_received = df_received.unionByName(unexpected).unionByName(duplicates)

    report = validate_external_dataframe(
        df_sent=df_sent,
        df_received=df_received,
        key_cols=["customer_id", "reference_date"],
        date_col="reference_date",
        score_col="bureau_score",
    )
    print(report["matching"])
    report = validate_external_dataframe(
        df_sent=df_sent,
        df_received=df_received,
        key_cols=["customer_id", "reference_date"],
        date_col="reference_date",
        score_col="bureau_score",
        baseline="202601",
    )
    print("Baseline:", report["baseline"], "Cutoffs:", report["cutoffs"])
    spark.stop()


if __name__ == "__main__":
    main()
