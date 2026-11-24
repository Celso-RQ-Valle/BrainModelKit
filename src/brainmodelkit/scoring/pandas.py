"""Bounded linear score display, independent of probability calibration."""

import numpy as np
import pandas as pd

from brainmodelkit._probability import ScoringConfig, outputs
from brainmodelkit._probability_pandas import numeric, rounded


class ScoreAligner(ScoringConfig):
    """Fixed scale endpoints, optional half-even decimal rounding."""

    def score_event(self, probabilities) -> np.ndarray:
        p = numeric(probabilities, self.invalid_policy, (0, 1))
        fraction = 1 - p if self.higher_score_means == "lower_event_probability" else p
        return np.clip(
            self.score_min + (self.score_max - self.score_min) * fraction,
            self.score_min,
            self.score_max,
        )

    def transform(
        self,
        df: pd.DataFrame,
        input_col: str = "score",
        *,
        output_col: str = "business_score",
        unrounded_col: str | None = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        raw_col = (
            (unrounded_col or output_col + "_unrounded")
            if self.decimals is not None
            else None
        )
        if unrounded_col is not None and self.decimals is None:
            raise ValueError("unrounded_col applies only when decimals is configured")
        outputs(df.columns, [output_col, raw_col], overwrite)
        p = numeric(df[input_col], self.invalid_policy, (0, 1))
        score = self.score_event(1 - p if self.input_probability == "non_event" else p)
        result = df.copy()
        result[output_col] = np.clip(
            rounded(score, self.decimals), self.score_min, self.score_max
        )
        if raw_col is not None:
            result[raw_col] = score
        return result

    def inverse_transform(
        self,
        df: pd.DataFrame,
        input_col: str = "business_score",
        *,
        output_col: str = "event_probability",
        output_probability: str = "event",
        overwrite: bool = False,
    ) -> pd.DataFrame:
        from brainmodelkit._probability import choice

        choice(output_probability, ("event", "non_event"), "output_probability")
        outputs(df.columns, [output_col], overwrite)
        score = numeric(
            df[input_col], self.invalid_policy, (self.score_min, self.score_max)
        )
        p = (score - self.score_min) / (self.score_max - self.score_min)
        if self.higher_score_means == "lower_event_probability":
            p = 1 - p
        result = df.copy()
        result[output_col] = p if output_probability == "event" else 1 - p
        return result
