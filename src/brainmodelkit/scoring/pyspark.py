"""Native SQL fixed-range linear score alignment."""

from brainmodelkit._probability import ScoringConfig, choice, outputs
from brainmodelkit._probability_spark import column, numeric


class ScoreAligner(ScoringConfig):
    def transform(
        self,
        df,
        input_col: str = "score",
        *,
        output_col: str = "business_score",
        unrounded_col: str | None = None,
        overwrite: bool = False,
    ):
        from pyspark.sql import functions as F

        raw_col = (
            (unrounded_col or output_col + "_unrounded")
            if self.decimals is not None
            else None
        )
        if unrounded_col is not None and self.decimals is None:
            raise ValueError("unrounded_col applies only when decimals is configured")
        outputs(df.columns, [output_col, raw_col], overwrite)
        p = numeric(df, input_col, self.invalid_policy, (0, 1))
        if self.input_probability == "non_event":
            p = 1 - p
        fraction = 1 - p if self.higher_score_means == "lower_event_probability" else p
        score = self.score_min + (self.score_max - self.score_min) * fraction
        displayed = (
            F.bround(score, self.decimals) if self.decimals is not None else score
        )
        displayed = F.when(
            score.isNotNull(),
            F.greatest(
                F.lit(self.score_min), F.least(F.lit(self.score_max), displayed)
            ),
        )
        additions = {output_col: displayed}
        if raw_col is not None:
            additions[raw_col] = score
        return df.select(
            *[
                additions.pop(name).alias(name) if name in additions else column(name)
                for name in df.columns
            ],
            *[value.alias(name) for name, value in additions.items()],
        )

    def inverse_transform(
        self,
        df,
        input_col: str = "business_score",
        *,
        output_col: str = "event_probability",
        output_probability: str = "event",
        overwrite: bool = False,
    ):
        choice(output_probability, ("event", "non_event"), "output_probability")
        outputs(df.columns, [output_col], overwrite)
        score = numeric(
            df, input_col, self.invalid_policy, (self.score_min, self.score_max)
        )
        p = (score - self.score_min) / (self.score_max - self.score_min)
        if self.higher_score_means == "lower_event_probability":
            p = 1 - p
        return df.withColumn(output_col, p if output_probability == "event" else 1 - p)
