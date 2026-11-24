"""Distributed logistic fitting and native SQL probability transformations."""

from brainmodelkit._probability import (
    CalibrationConfig,
    CapabilityError,
    check_fit_summary,
    outputs,
)
from brainmodelkit._probability_spark import (
    column,
    labels_and_weights,
    numeric,
    sigmoid,
    transformed,
)


class LogisticCalibrator(CalibrationConfig):
    """Portable fixed parameters; fitting separately requires Spark ML."""

    def transform(
        self,
        df,
        input_col: str = "score",
        *,
        output_col: str = "calibrated_probability",
        event_col: str | None = None,
        non_event_col: str | None = None,
        overwrite: bool = False,
    ):
        self._ready()
        outputs(df.columns, [output_col, event_col, non_event_col], overwrite)
        event = sigmoid(self.intercept + self.slope * transformed(self, df, input_col))
        # Evaluate against original input, including when overwriting columns.
        additions = {
            output_col: event if self.output_probability == "event" else 1 - event
        }
        if event_col is not None:
            additions[event_col] = event
        if non_event_col is not None:
            additions[non_event_col] = 1 - event
        return df.select(
            *[
                additions.pop(name).alias(name) if name in additions else column(name)
                for name in df.columns
            ],
            *[value.alias(name) for name, value in additions.items()],
        )

    def fit(
        self,
        df,
        input_col: str = "score",
        target_col: str = "target",
        *,
        weight_col: str | None = None,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ):
        from pyspark.sql import functions as F

        self._fit_options(max_iter, tol)
        x = transformed(self, df, input_col)
        y, w = labels_and_weights(
            df, target_col, self.event_label, weight_col, self.invalid_policy
        )
        data = df.select(x.alias("x"), y.alias("y"), w.alias("w")).where(
            "x IS NOT NULL AND y IS NOT NULL AND w > 0"
        )
        summary = data.agg(
            F.count("*").alias("n"),
            *[
                op(F.when(F.col("y") == label, F.col("x"))).alias(f"{kind}{label}")
                for label in (0, 1)
                for kind, op in (("lo", F.min), ("hi", F.max))
            ],
        ).first()
        check_fit_summary(summary.n, summary.lo0, summary.hi0, summary.lo1, summary.hi1)
        try:
            from pyspark.ml.classification import LogisticRegression
            from pyspark.ml.feature import VectorAssembler

            features = VectorAssembler(inputCols=["x"], outputCol="features").transform(
                data
            )
            fitted = LogisticRegression(
                featuresCol="features",
                labelCol="y",
                weightCol="w",
                regParam=0.0,
                elasticNetParam=0.0,
                standardization=True,
                maxIter=max_iter,
                tol=tol,
                family="binomial",
            ).fit(features)
            iterations = fitted.summary.totalIterations
            intercept, slope = float(fitted.intercept), float(fitted.coefficients[0])
        except Exception as exc:
            raise CapabilityError(
                "Spark ML calibration fitting failed in this runtime. Manual "
                "LogisticCalibrator(intercept=..., slope=...) uses SQL only "
                "and remains available. Original error: " + str(exc)
            ) from exc
        if iterations >= max_iter:
            raise RuntimeError(
                "Calibration reached max_iter; increase it or inspect the data"
            )
        return self._fitted(
            intercept,
            slope,
            {
                "solver": "Spark ML L-BFGS",
                "regularization": 0.0,
                "converged": True,
                "iterations": iterations,
                "observations": summary.n,
                "tolerance": tol,
            },
        )


def calibration_diagnostics(
    df,
    probability_col: str = "score",
    target_col: str = "target",
    *,
    event_label=1,
    input_probability: str = "event",
    weight_col: str | None = None,
    n_bins: int = 10,
    epsilon: float = 1e-12,
    invalid_policy: str = "raise",
) -> dict:
    """Distributed aggregates; collect one scalar summary, return Spark reliability."""
    from pyspark.sql import functions as F

    config = LogisticCalibrator(
        0,
        1,
        input_probability=input_probability,
        epsilon=epsilon,
        invalid_policy=invalid_policy,
        event_label=event_label,
    )
    if type(n_bins) is not int or n_bins < 1:
        raise ValueError("n_bins must be positive")
    p = numeric(df, probability_col, invalid_policy, (0, 1))
    if config.input_probability == "non_event":
        p = 1 - p
    y, w = labels_and_weights(df, target_col, event_label, weight_col, invalid_policy)
    data = df.select(p.alias("p"), y.alias("y"), w.alias("w"))
    valid = F.col("p").isNotNull() & F.col("y").isNotNull() & (F.col("w") > 0)
    p, y, w = F.col("p"), F.col("y"), F.col("w")
    safe = F.greatest(F.lit(epsilon), F.least(F.lit(1 - epsilon), p))
    summary = data.agg(
        F.sum(F.when(valid, w * (p - y) ** 2)).alias("brier"),
        F.sum(F.when(valid, w * (-y * F.log(safe) - (1 - y) * F.log1p(-safe)))).alias(
            "loss"
        ),
        F.sum(F.when(valid, w)).alias("weight"),
        F.sum(valid.cast("long")).alias("n"),
        F.count("*").alias("total"),
    ).first()
    if not summary.n:
        raise ValueError("No valid positive-weight observations")
    table = (
        data.where(valid)
        .withColumn("bin", F.least((p * n_bins).cast("int"), F.lit(n_bins - 1)))
        .groupBy("bin")
        .agg(
            F.count("*").alias("count"),
            F.sum(w).alias("weight_sum"),
            (F.sum(p * w) / F.sum(w)).alias("mean_probability"),
            (F.sum(y * w) / F.sum(w)).alias("event_rate"),
        )
        .orderBy("bin")
    )
    return {
        "brier_score": summary.brier / summary.weight,
        "log_loss": summary.loss / summary.weight,
        "count": summary.n,
        "omitted_count": summary.total - summary.n,
        "event_label": event_label,
        "reliability": table,
    }


def compare_calibration(
    df,
    before_col: str,
    after_col: str,
    target_col: str = "target",
    *,
    before_probability: str = "event",
    after_probability: str = "event",
    **kwargs,
) -> dict:
    return {
        "before": calibration_diagnostics(
            df, before_col, target_col, input_probability=before_probability, **kwargs
        ),
        "after": calibration_diagnostics(
            df, after_col, target_col, input_probability=after_probability, **kwargs
        ),
    }
