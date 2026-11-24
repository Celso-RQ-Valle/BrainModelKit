"""Manual and unpenalized fitted logistic calibration for pandas."""

import numpy as np
import pandas as pd

from brainmodelkit._probability import CalibrationConfig, check_fit_summary, outputs
from brainmodelkit._probability_pandas import (
    labels_and_weights,
    numeric,
    sigmoid,
    transformed,
)


class LogisticCalibrator(CalibrationConfig):
    """Compute sigmoid(intercept + slope * input), normalized to the event.

    No parameters means unfitted. Supplying both parameters selects manual mode.
    invalid_policy='null' maps invalid inputs to NaN and omits them during fitting.
    """

    def predict_event(self, values) -> np.ndarray:
        self._ready()
        x = transformed(self, values)
        with np.errstate(over="ignore"):
            return sigmoid(self.intercept + self.slope * x)

    def transform(
        self,
        df: pd.DataFrame,
        input_col: str = "score",
        *,
        output_col: str = "calibrated_probability",
        event_col: str | None = None,
        non_event_col: str | None = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        outputs(df.columns, [output_col, event_col, non_event_col], overwrite)
        event = self.predict_event(df[input_col])
        result = df.copy()
        result[output_col] = event if self.output_probability == "event" else 1 - event
        if event_col is not None:
            result[event_col] = event
        if non_event_col is not None:
            result[non_event_col] = 1 - event
        return result

    def fit(
        self,
        df: pd.DataFrame,
        input_col: str = "score",
        target_col: str = "target",
        *,
        weight_col: str | None = None,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ):
        """Fit only on explicitly supplied held-out/OOF predictions, never a model."""
        from scipy.optimize import minimize

        self._fit_options(max_iter, tol)
        x = transformed(self, df[input_col])
        y, w, labeled = labels_and_weights(
            df, target_col, self.event_label, weight_col, self.invalid_policy
        )
        valid = np.isfinite(x) & labeled & (w > 0)
        x, y, w = x[valid], y[valid], w[valid]
        ranges = [
            float(op(x[y == label])) if np.any(y == label) else None
            for label in (0, 1)
            for op in (np.min, np.max)
        ]
        check_fit_summary(len(x), *ranges)
        w = w / w.sum()
        center = float(np.sum(w * x))
        scale = float(np.sqrt(np.sum(w * (x - center) ** 2)))
        design = np.column_stack([np.ones(len(x)), (x - center) / scale])

        def objective(beta):
            z = design @ beta
            return float(np.sum(w * (np.logaddexp(0, z) - y * z))), design.T @ (
                w * (sigmoid(z) - y)
            )

        fitted = minimize(
            objective,
            np.zeros(2),
            jac=True,
            method="BFGS",
            options={"gtol": tol, "maxiter": max_iter},
        )
        if not fitted.success:
            raise RuntimeError(f"Calibration did not converge: {fitted.message}")
        slope = float(fitted.x[1] / scale)
        return self._fitted(
            float(fitted.x[0] - slope * center),
            slope,
            {
                "solver": "scipy.BFGS",
                "regularization": 0.0,
                "converged": True,
                "iterations": int(fitted.nit),
                "observations": len(x),
                "omitted_observations": len(df) - len(x),
                "objective": float(fitted.fun),
                "tolerance": tol,
                "target_classes": [
                    v.item() if hasattr(v, "item") else v
                    for v in df[target_col].dropna().unique()
                ],
            },
        )


def calibration_diagnostics(
    df: pd.DataFrame,
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
    """Weighted Brier/log loss and equal-width reliability bins (nonempty bins)."""
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
    p = numeric(df[probability_col], invalid_policy, (0, 1))
    if config.input_probability == "non_event":
        p = 1 - p
    y, w, labeled = labels_and_weights(
        df, target_col, event_label, weight_col, invalid_policy
    )
    valid = np.isfinite(p) & labeled & (w > 0)
    p, y, w = p[valid], y[valid], w[valid]
    if not len(p):
        raise ValueError("No valid positive-weight observations")
    safe = np.clip(p, epsilon, 1 - epsilon)
    rows = pd.DataFrame(
        {
            "bin": np.minimum((p * n_bins).astype(int), n_bins - 1),
            "weight": w,
            "prediction": p * w,
            "event": y * w,
        }
    )
    table = (
        rows.groupby("bin", sort=True)
        .agg(
            count=("weight", "size"),
            weight_sum=("weight", "sum"),
            mean_probability=("prediction", "sum"),
            event_rate=("event", "sum"),
        )
        .reset_index()
    )
    table["mean_probability"] /= table.weight_sum
    table["event_rate"] /= table.weight_sum
    return {
        "brier_score": float(np.average((p - y) ** 2, weights=w)),
        "log_loss": float(
            np.average(-y * np.log(safe) - (1 - y) * np.log1p(-safe), weights=w)
        ),
        "count": len(p),
        "omitted_count": len(df) - len(p),
        "event_label": event_label,
        "reliability": table,
    }


def compare_calibration(
    df: pd.DataFrame,
    before_col: str,
    after_col: str,
    target_col: str = "target",
    *,
    before_probability: str = "event",
    after_probability: str = "event",
    **kwargs,
) -> dict:
    """Evaluate two probability columns using explicit probability semantics."""
    return {
        "before": calibration_diagnostics(
            df, before_col, target_col, input_probability=before_probability, **kwargs
        ),
        "after": calibration_diagnostics(
            df, after_col, target_col, input_probability=after_probability, **kwargs
        ),
    }
