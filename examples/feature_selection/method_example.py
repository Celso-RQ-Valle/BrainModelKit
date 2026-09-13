"""Run any selector: --backend pandas|spark --method boruta (or another method).

Use --quick for a small smoke-test budget; tentative Boruta outcomes are expected.
Algorithm guides: docs/feature_selection/<method>.md.
"""

import argparse
import importlib
import os
import sys


def run(backend, method, quick=False):
    fs = importlib.import_module(f"brainmodelkit.feature_selection.{backend}")
    if method not in fs.__all__:
        raise ValueError(f"Unknown method: {method}. Choose one of {fs.__all__}")
    spark = None
    features = ["signal", "noise", "constant"]
    if backend == "pandas":
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            {
                "signal": np.tile([0.0, 1.0], 120),
                "noise": rng.random(240),
                "constant": 1.0,
                "target": np.tile([0, 1], 120),
            }
        )
        train, validation = data.iloc[:180].copy(), data.iloc[180:].copy()
        if method == "chi_square":
            train["noise"] = (train["noise"] > 0.5).astype(int)
    else:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        spark = (
            SparkSession.builder.master("local[2]")
            .appName("SelectionExample")
            .getOrCreate()
        )
        data = (
            spark.range(240)
            .withColumn("target", (F.col("id") % 2).cast("int"))
            .withColumn("signal", F.col("target").cast("double"))
            .withColumn("noise", F.rand(42))
            .withColumn("constant", F.lit(1.0))
            .withColumn("fold", (F.floor(F.col("id") / 2) % 3).cast("int"))
        )
        train, validation = data.filter("id < 180"), data.filter("id >= 180")
        if method == "chi_square":
            train = train.withColumn("noise", (F.col("noise") > 0.5).cast("int"))
    try:
        selector = getattr(fs, method)
        if method in {
            "completeness",
            "variance_filter",
            "cardinality",
            "correlation_filter",
        }:
            result = selector(train, features)
        elif method == "permutation_importance_selection":
            model = fs.feature_importance_selection(
                train, "target", features, model="decision_tree"
            ).model
            result = selector(
                model, validation, "target", features, n_repeats=1 if quick else 5
            )
        elif method == "rfe":
            result = selector(
                "example",
                "target",
                train,
                validation,
                features,
                n_feat_final=1,
                output_dir="feature_selection_runs",
            )
        else:
            options = {}
            if method in {"rfecv", "sequential_selection"}:
                options.update(cv=3, model="decision_tree")
                if backend == "pyspark":
                    options["fold_col"] = "fold"
            elif method == "stability_selection":
                options.update(
                    n_iterations=2 if quick else 10,
                    selector_kwargs={"model": "decision_tree", "top_k": 1},
                )
            elif method == "boruta":
                options.update(
                    n_estimators=10 if quick else 30, max_iter=2 if quick else 30
                )
            elif method in {"mutual_information", "feature_importance_selection"}:
                options["top_k"] = 1
            result = selector(train, "target", features, **options)
        print("Selected:", result.selected_features)
        print("Rejected:", result.rejected_features)
        print("Diagnostics:", result.feature_table)
        return result.selected_features
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["pandas", "spark"], default="pandas")
    parser.add_argument("--method", default="completeness")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run("pyspark" if args.backend == "spark" else "pandas", args.method, args.quick)
