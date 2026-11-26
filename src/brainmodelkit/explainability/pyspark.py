"""Worker-local SHAP for exportable Spark models; explicit bounded local mode."""

from brainmodelkit._probability import CapabilityError, choice
from brainmodelkit._probability_spark import column
from brainmodelkit.persistence import resolve_model
from brainmodelkit.prediction.pyspark import PredictionPipeline as PredictionPipeline
from brainmodelkit.prediction.spark_adapters import export_predictor, restore_predictor


class ExplanationResult:
    """Distributed wide explanation frame with all original-feature contributions."""

    def __init__(self, frame, metadata: dict, features: list[str], keys: list[str]):
        self.frame, self.metadata, self.features, self.keys = (
            frame,
            metadata,
            features,
            keys,
        )

    def to_frame(self):
        return self.frame

    def global_importance(self):
        from pyspark.sql import functions as F

        summary = self.frame.agg(
            *[
                F.avg(F.abs(column("contribution::" + name))).alias(name)
                for name in self.features
            ]
        )
        return (
            summary.select(
                F.explode(
                    F.array(
                        *[
                            F.struct(
                                F.lit(name).alias("feature"),
                                column(name).alias("mean_absolute_contribution"),
                            )
                            for name in self.features
                        ]
                    )
                ).alias("item")
            )
            .select("item.*")
            .orderBy(F.desc("mean_absolute_contribution"))
        )

    def aggregate_groups(self, groups: dict[str, str]):
        from pyspark.sql import functions as F

        if set(groups) - set(self.features):
            raise ValueError("Unknown grouped features")
        grouped = {}
        for feature in self.features:
            group = groups.get(feature, feature)
            grouped[group] = grouped.get(group, F.lit(0.0)) + column(
                "contribution::" + feature
            )
        return self.frame.select(
            *[column(name) for name in self.keys],
            *[value.alias("group::" + name) for name, value in grouped.items()],
        )

    def top_contributors(self, top_k: int = 5):
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be positive")
        rows = self.frame.select(
            *[column(name) for name in self.keys],
            F.explode(
                F.array(
                    *[
                        F.struct(
                            F.lit(name).alias("feature"),
                            column("contribution::" + name).alias("contribution"),
                            column("value::" + name).alias("value"),
                        )
                        for name in self.features
                    ]
                )
            ).alias("item"),
        ).select(*[column(name) for name in self.keys], "item.*")
        rows = rows.where("contribution != 0").withColumn(
            "direction",
            F.when(F.col("contribution") > 0, "positive").otherwise("negative"),
        )
        window = Window.partitionBy(
            *[column(name) for name in self.keys], "direction"
        ).orderBy(F.desc(F.abs(F.col("contribution"))), "feature")
        return rows.withColumn("rank", F.row_number().over(window)).where(
            F.col("rank") <= top_k
        )

    def local(self, max_rows: int = 1):
        """Explicit bounded collection for waterfall/summary; filter frame first."""
        import numpy as np
        import pandas as pd

        from ._result import ExplanationResult as LocalResult

        if type(max_rows) is not int or max_rows < 1:
            raise ValueError("max_rows must be positive")
        frame = self.frame.limit(max_rows + 1).toPandas()
        if len(frame) > max_rows:
            raise ValueError(
                "Result exceeds max_rows; filter by observation keys first"
            )
        values = pd.DataFrame({name: frame["value::" + name] for name in self.features})
        contributions = pd.DataFrame(
            {name: frame["contribution::" + name] for name in self.features}
        )
        return LocalResult(
            frame.prediction.to_numpy(),
            frame.base_value.to_numpy(),
            contributions,
            values,
            frame[self.keys],
            self.metadata,
            frame.residual.to_numpy(),
            frame.tolerance.to_numpy(),
            np.asarray(frame.rounded_prediction)
            if "rounded_prediction" in frame
            else None,
        )


class ModelExplainer:
    """Explicit distributed mapInPandas or bounded local execution.

    Spark models are exported on the driver, then reconstructed once per partition.
    No nested Spark calls occur in workers. Numeric scalar features only.
    """

    def __init__(
        self,
        model,
        background,
        feature_cols: list[str] | None = None,
        *,
        key_cols: list[str],
        execution: str = "distributed",
        max_background: int = 100,
        max_rows: int = 1000,
        seed: int = 42,
        **options,
    ):
        from pyspark.sql import functions as F

        from brainmodelkit._probability_spark import numeric
        from brainmodelkit.calibration.pandas import LogisticCalibrator
        from brainmodelkit.scoring.pandas import ScoreAligner

        from .pandas import ModelExplainer as LocalExplainer

        choice(execution, ("distributed", "local"), "execution")
        if any(
            type(value) is not int or value < 1 for value in (max_background, max_rows)
        ):
            raise ValueError("max_background and max_rows must be positive")
        if not key_cols or len(set(key_cols)) != len(key_cols):
            raise ValueError(
                "Spark explanations require unique key_cols (composite keys supported)"
            )
        resolved = resolve_model(
            model,
            feature_cols=feature_cols,
            class_labels=options.pop("class_labels", None),
        )
        if resolved.class_labels is not None and resolved.class_labels != [0, 1]:
            raise ValueError("Spark probability vectors require class indices [0, 1]")
        self.features, self.keys = resolved.feature_cols or [], list(key_cols)
        if (
            not self.features
            or len(set(self.features)) != len(self.features)
            or set(self.keys) & set(self.features)
        ):
            raise ValueError("Provide unique feature_cols separate from key_cols")
        if isinstance(model, PredictionPipeline):
            if (
                model.probability_col is not None
                and model.probability_col != model.adapter.probability_col
            ):
                raise CapabilityError(
                    "Export explains the fitted classifier probability column only"
                )
            if model.feature_cols != self.features:
                raise ValueError("Feature order differs from prediction pipeline")
            for name in ("event_label", "calibrator", "score_aligner"):
                if name in options:
                    raise ValueError(
                        "Configure components on PredictionPipeline only once"
                    )
                options[name] = getattr(model, name)
        if isinstance(model, PredictionPipeline) and hasattr(
            model, "feature_dictionary"
        ):
            options.setdefault("feature_dictionary", model.feature_dictionary)
        if set(self.features) - set(background.columns) or len(
            background.columns
        ) != len(set(background.columns)):
            raise ValueError("Missing features or duplicate background columns")
        self.schema = [
            (name, background.schema[name].dataType) for name in self.features
        ]
        for name in self.features:
            numeric(background, name)
        self.spec = export_predictor(resolved.model, self.features)
        # Content-hash sample, bounded before collection; keys are not features.
        selected = background.orderBy(
            F.xxhash64(F.lit(seed), *[column(name) for name in self.features]),
            *[column(name) for name in self.features],
        ).limit(max_background)
        self.background = selected.select(
            *[column(name).cast("double").alias(name) for name in self.features]
        ).toPandas()
        if self.spec["kind"] == "synapseml":
            import numpy as np
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.functions import vector_to_array

            fitted = resolved.model
            ref = selected.select(*self.features)
            estimator = fitted.stages[-1] if hasattr(fitted, "stages") else fitted
            if fitted is estimator:
                ref = VectorAssembler(
                    inputCols=self.features, outputCol=estimator.getFeaturesCol()
                ).transform(ref)
            pairs = (
                fitted.transform(ref)
                .select(
                    *self.features,
                    vector_to_array(column(estimator.getProbabilityCol()))[1].alias(
                        "__expected"
                    ),
                )
                .toPandas()
            )
            if not np.allclose(
                restore_predictor(self.spec, self.features).predict(
                    pairs[self.features]
                ),
                pairs["__expected"],
                rtol=1e-6,
                atol=1e-8,
            ):
                raise CapabilityError(
                    "SynapseML native export differs from Spark probabilities"
                )
        for name, cls in (
            ("calibrator", LogisticCalibrator),
            ("score_aligner", ScoreAligner),
        ):
            if options.get(name) is not None:
                options[name] = cls.from_dict(options[name].to_dict())
        self.options = {
            **options,
            "key_cols": self.keys,
            "max_background": max_background,
            "max_rows": max_rows,
            "seed": seed,
            "class_labels": [0, 1],
        }
        self.execution, self.max_rows = execution, max_rows
        self.prototype = LocalExplainer(
            restore_predictor(self.spec, self.features),
            self.background,
            self.features,
            **self.options,
        )
        self.prototype.pipeline.feature_space = getattr(
            model, "feature_space", "original"
        )
        self.prototype._initialize()

    def explain(self, df):
        from pyspark.sql import functions as F
        from pyspark.sql.types import BooleanType, DoubleType, StructField, StructType

        from brainmodelkit._probability_spark import numeric

        from .pandas import ModelExplainer as LocalExplainer

        if len(df.columns) != len(set(df.columns)) or set(
            self.features + self.keys
        ) - set(df.columns):
            raise ValueError("Missing features, missing keys, or duplicate columns")
        for name, dtype in self.schema:
            if df.schema[name].dataType != dtype:
                raise TypeError("Feature dtypes differ from background")
            numeric(df, name)
        if (
            df.where(
                F.greatest(
                    *[column(name).isNull().cast("int") for name in self.keys], F.lit(0)
                )
                > 0
            )
            .limit(1)
            .count()
        ):
            raise ValueError("Observation keys must not be null")
        if (
            df.groupBy(*[column(name) for name in self.keys])
            .count()
            .where("count > 1")
            .limit(1)
            .count()
        ):
            raise ValueError("Composite observation keys must be unique")
        selected = df.select(
            *[column(name) for name in self.keys],
            *[column(name).cast("double").alias(name) for name in self.features],
        )
        if self.execution == "local":
            local = selected.limit(self.max_rows + 1).toPandas()
            return self.prototype.explain(local)
        fields = [df.schema[name] for name in self.keys]
        for name in ("prediction", "base_value", "residual", "tolerance"):
            fields.append(StructField(name, DoubleType(), False))
        fields.append(StructField("reconstruction_ok", BooleanType(), False))
        if (
            self.prototype.target == "business_score"
            and self.prototype.pipeline.score_aligner.decimals is not None
        ):
            fields += [
                StructField(name, DoubleType(), False)
                for name in ("rounded_prediction", "rounding_difference")
            ]
        for name in self.features:
            fields += [
                StructField("contribution::" + name, DoubleType(), False),
                StructField("value::" + name, DoubleType(), False),
            ]
        spec, features, background, options = (
            self.spec,
            self.features,
            self.background,
            self.options,
        )

        # Capture portable state only, never self/prototype/JVM model/SparkSession.
        def batches(iterator):
            explainer = LocalExplainer(
                restore_predictor(spec, features), background, features, **options
            )
            for batch in iterator:
                for start in range(0, len(batch), options["max_rows"]):
                    yield explainer.explain(
                        batch.iloc[start : start + options["max_rows"]]
                    ).to_frame()

        result = selected.mapInPandas(batches, StructType(fields))
        metadata = {
            **self.prototype.pipeline.metadata(),
            "feature_space": self.prototype.pipeline.feature_space,
            "target": self.prototype.target,
            "units": "score points"
            if self.prototype.target == "business_score"
            else self.prototype.pipeline.adapter.raw_units
            if self.prototype.target == "raw"
            else "probability (0 to 1)",
            "reference": self.prototype.reference,
            "method": self.prototype.method,
            "max_evals": self.prototype.max_evals,
            "seed": self.prototype.seed,
            "feature_dictionary": self.prototype.feature_dictionary,
            "execution": "distributed mapInPandas",
            "adapter": self.spec["kind"],
            "atol": self.prototype.atol,
            "rtol": self.prototype.rtol,
        }
        return ExplanationResult(result, metadata, self.features, self.keys)
