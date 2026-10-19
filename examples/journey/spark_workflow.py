"""Run: python examples/journey/spark_workflow.py [--save-dir PATH]."""

import argparse
import os
import sys

from sample_data import make_rows

from brainmodelkit.cube_analysis.pyspark import calculate_metrics, calculate_ntile
from brainmodelkit.feature_selection.pyspark import variance_filter
from brainmodelkit.metrics.pyspark import auc_gini, ks_ntile, risk_table
from brainmodelkit.persistence import load_model
from brainmodelkit.pyspark import build_features
from brainmodelkit.training.pyspark import train_model


def main(save_dir=None):
    from pyspark.ml import PipelineModel
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("ModelingJourney")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        base_rows, spine_rows = make_rows()
        df_base = spark.createDataFrame(
            base_rows,
            "account_id string, observation_date date, revenue double, balance double",
        )
        df_spine = spark.createDataFrame(
            spine_rows,
            "account_id string, reference_date date, segment string, target int",
        )
        enriched = build_features(
            df_base,
            df_spine,
            key_cols="account_id",
            value_cols=["revenue", "balance"],
            reference_date_col="reference_date",
            windows=[3],
            lags=[1],
            aggregations=["mean", "sum", "count"],
        )
        candidate_features = [c for c in enriched.columns if c not in df_spine.columns]
        enriched = enriched.fillna(0, subset=candidate_features).cache()
        train_df = enriched.filter("reference_date < '2026-05-01'")
        validation_df = enriched.filter(
            "reference_date >= '2026-05-01' AND reference_date < '2026-06-01'"
        )
        oot_df = enriched.filter("reference_date >= '2026-06-01'")
        selected_features = variance_filter(
            train_df, feature_cols=candidate_features
        ).selected_features
        assert selected_features
        trials = []
        for reg in (0.01, 0.1):
            baseline = train_model(
                "development",
                "target",
                selected_features,
                train_df,
                validation_df,
                model="logistic_regression",
                model_params={"regParam": reg, "maxIter": 10},
                save_model_to="none",
                n_tiles=4,
            )
            trials.append((baseline.metrics["oot_auc"], reg))
        best_reg = max(trials, key=lambda trial: trial[0])[1]
        development_df = train_df.unionByName(validation_df)
        result = train_model(
            "final_spark",
            "target",
            selected_features,
            development_df,
            oot_df,
            model="logistic_regression",
            model_params={"regParam": best_reg, "maxIter": 10},
            save_model_to="folder" if save_dir else "none",
            output_dir=save_dir or "training_runs",
            n_tiles=4,
        )
        predictions = result.oot_predictions.cache()
        print(result.metrics)
        ks_ntile(dataframe=predictions, n_tiles=4).show()
        auc_gini(df=predictions, n_tiles=4).show()
        risk_table(df=predictions, n_tiles=4).show()
        calculate_metrics(df=predictions, group_columns=["segment"], n_tiles=4).show()
        calculate_ntile(df=predictions, group_columns=["segment"], n_tiles=4).show()
        fitted = (
            load_model(result.model_uri, format="spark", model_class=PipelineModel)
            if save_dir
            else result.model
        )
        new_data = oot_df.drop("target")
        scored = fitted.transform(new_data).withColumn(
            "score", vector_to_array("probability")[1]
        )
        assert scored.count() == 40
        assert (
            scored.filter(
                F.col("score").isNull() | ~F.col("score").between(0, 1)
            ).count()
            == 0
        )
        print("Spark journey passed; model URI:", result.model_uri)
        predictions.unpersist()
        enriched.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir")
    main(parser.parse_args().save_dir)
