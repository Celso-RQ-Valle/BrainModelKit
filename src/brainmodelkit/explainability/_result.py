"""Complete local attributions, deterministic reports, and optional plotting."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ExplanationResult:
    predictions: np.ndarray
    base_values: np.ndarray
    contributions: pd.DataFrame
    feature_values: pd.DataFrame
    keys: pd.DataFrame
    metadata: dict[str, Any]
    residuals: np.ndarray
    tolerances: np.ndarray
    rounded_predictions: np.ndarray | None = None

    def to_frame(self) -> pd.DataFrame:
        """Wide frame, preserving input index, keys, and every contribution."""
        frame = self.keys.copy()
        frame["prediction"] = self.predictions
        frame["base_value"] = self.base_values
        frame["residual"] = self.residuals
        frame["tolerance"] = self.tolerances
        frame["reconstruction_ok"] = np.abs(self.residuals) <= self.tolerances
        if self.rounded_predictions is not None:
            frame["rounded_prediction"] = self.rounded_predictions
            frame["rounding_difference"] = self.rounded_predictions - self.predictions
        for name in self.contributions:
            frame[f"contribution::{name}"] = self.contributions[name].to_numpy()
            frame[f"value::{name}"] = self.feature_values[name].to_numpy()
        return frame

    def top_contributors(
        self, observation: int = 0, top_k: int = 5
    ) -> dict[str, pd.Series]:
        """Positive/negative refers to the declared output, not risk in general."""
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be positive")
        row = self.contributions.iloc[observation]
        return {
            "positive": row[row > 0]
            .sort_values(ascending=False, kind="stable")
            .head(top_k),
            "negative": row[row < 0].sort_values(kind="stable").head(top_k),
        }

    def global_importance(self) -> pd.Series:
        """Mean absolute attribution in this result's output units/population."""
        return (
            self.contributions.abs().mean().sort_values(ascending=False, kind="stable")
        )

    def aggregate_groups(self, groups: dict[str, str] | None = None) -> pd.DataFrame:
        """Sum contributions; unspecified features retain their own group."""
        dictionary = self.metadata.get("feature_dictionary", {})
        groups = groups or {
            name: values.get("group", name) for name, values in dictionary.items()
        }
        unknown = set(groups) - set(self.contributions)
        if unknown:
            raise ValueError(f"Unknown grouped features: {sorted(unknown)}")
        result = pd.DataFrame(index=self.contributions.index)
        for name in self.contributions:
            group = groups.get(name, name)
            result[group] = result.get(group, 0) + self.contributions[name]
        return result

    def summary(self, observation: int = 0, top_k: int = 3) -> str:
        dictionary = self.metadata.get("feature_dictionary", {})
        units = self.metadata["units"]
        lines = [
            f"Prediction {self.predictions[observation]:.6g} {units}; "
            f"reference {self.base_values[observation]:.6g}."
        ]
        for direction, values in self.top_contributors(observation, top_k).items():
            for feature, contribution in values.items():
                name = dictionary.get(feature, {}).get("display_name", feature)
                value = self.feature_values.iloc[observation][feature]
                lines.append(
                    f"{name} (observed {value!s}): {contribution:+.6g} "
                    f"{units} ({direction} contribution)."
                )
        lines.append(
            "Attributions describe this model relative to its reference, "
            "not causal effects."
        )
        return "\n".join(lines)

    def waterfall(self, observation: int = 0, top_k: int = 10, ax=None):
        """Additive chart in unrounded output units; retain an other-features term."""
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be positive")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Install brainmodelkit[plots] for waterfall plots"
            ) from exc
        row = self.contributions.iloc[observation]
        selected = (
            row.abs().sort_values(ascending=False, kind="stable").head(top_k).index
        )
        values = row.loc[selected].tolist()
        dictionary = self.metadata.get("feature_dictionary", {})
        names = [
            dictionary.get(name, {}).get("display_name", name) for name in selected
        ]
        if len(selected) < len(row):
            values.append(float(row.drop(selected).sum()))
            names.append("Other features")
        if ax is None:
            _, ax = plt.subplots()
        running = float(self.base_values[observation])
        for position, value in enumerate(values):
            ax.barh(
                position,
                abs(value),
                left=min(running, running + value),
                color="tab:red" if value > 0 else "tab:blue",
            )
            running += value
        ax.set_yticks(range(len(names)), names)
        ax.axvline(
            self.base_values[observation],
            color="gray",
            linestyle="--",
            label="Reference",
        )
        ax.axvline(self.predictions[observation], color="black", label="Prediction")
        ax.set_xlabel(self.metadata["units"])
        ax.legend()
        return ax
