"""Run with Java 17 and brainmodelkit[pyspark,explainability] installed.

python examples/probability_workflow_pyspark.py --output spark_probability_artifacts
Uses classic local Spark, not a claim of Databricks Serverless compatibility.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from brainmodelkit.calibration.pyspark import LogisticCalibrator, compare_calibration
from brainmodelkit.explainability.pyspark import ModelExplainer
from brainmodelkit.persistence import ArtifactSpec
from brainmodelkit.prediction.pyspark import PredictionPipeline
from brainmodelkit.scoring.pyspark import ScoreAligner
from brainmodelkit.training.pyspark import train_model


def main(output: Path, execution="distributed", save_model=False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("CreditCalibration")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    try:
        rng = np.random.default_rng(42)
        pdf = pd.DataFrame(
            {
                "utilization": rng.uniform(0, 1, 800),
                "late_payments": rng.poisson(1, 800).astype(float),
            }
        )
        features = list(pdf)
        p = 1 / (1 + np.exp(-(-2 + 2 * pdf.utilization + 0.4 * pdf.late_payments)))
        pdf["target"] = rng.binomial(1, p)
        pdf["customer_id"], pdf["month"] = np.arange(800), 1
        data = spark.createDataFrame(pdf)
        train = data.where("customer_id < 400")
        development = data.where("customer_id >= 400 AND customer_id < 500")
        calibration = data.where("customer_id >= 500 AND customer_id < 650")
        final = data.where("customer_id >= 650")
        trained = train_model(
            "credit_calibration",
            "target",
            features,
            train,
            development,
            model_params={"maxIter": 100},
            save_model_to="folder" if save_model else "none",
            output_dir=output,
        )
        source = (
            ArtifactSpec(
                path=trained.model_uri, format="spark", model_class=PipelineModel
            )
            if save_model
            else trained
        )
        base = PredictionPipeline(source, features)
        cal_predictions = base.transform(calibration)
        fitted = LogisticCalibrator(feature_order=features).fit(
            cal_predictions, "event_probability", "target"
        )
        # Independent manual transform; it does not certify empirical calibration.
        LogisticCalibrator(-0.1, 0.9).transform(
            cal_predictions, "event_probability", output_col="manual_probability"
        ).select("manual_probability").show(3)
        aligner = ScoreAligner(decimals=0, feature_order=features)
        workflow = PredictionPipeline(base, calibrator=fitted, score_aligner=aligner)
        predictions = workflow.transform(final).cache()
        reports = compare_calibration(
            predictions, "event_probability", "calibrated_event_probability"
        )
        print({name: report["brier_score"] for name, report in reports.items()})
        reports["after"]["reliability"].show()
        ScoreAligner(300, 850, input_probability="non_event").transform(
            predictions, "calibrated_non_event_probability", output_col="custom_score"
        ).select("business_score", "custom_score").show(3)
        explainer = ModelExplainer(
            workflow,
            train,
            key_cols=["customer_id", "month"],
            execution=execution,
            max_rows=3,
            target="business_score",
            method="permutation",
            max_background=10,
            max_evals=15,
            reference_id="synthetic-credit-training-v1",
        )
        explained = explainer.explain(predictions.where("customer_id < 653"))
        if execution == "distributed":
            explained.to_frame().show(truncate=False)
        else:
            print(explained.to_frame().to_string(index=False))
        # Explicit bounded local diagnostics for one customer, using the same reference.
        local = ModelExplainer(
            trained,
            train.select(*features),
            features,
            key_cols=["customer_id", "month"],
            calibrator=fitted,
            target="calibrated_event_probability",
            method="permutation",
            execution="local",
            max_rows=1,
            max_background=10,
            max_evals=15,
            reference_id="synthetic-credit-training-v1",
        )
        print(
            local.explain(
                final.select("customer_id", "month", *features).where(
                    "customer_id = 650"
                )
            ).summary()
        )
        fitted.save(output / "calibration.json")
        aligner.save(output / "scoring.json")
        restored = LogisticCalibrator.load(output / "calibration.json")
        assert restored.slope == fitted.slope
        predictions.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("spark_probability_artifacts")
    )
    parser.add_argument(
        "--execution", choices=["distributed", "local"], default="distributed"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save and reload the fitted Spark PipelineModel",
    )
    args = parser.parse_args()
    main(args.output, args.execution, args.save_model)
