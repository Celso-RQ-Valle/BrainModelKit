"""Pandas numeric validation shared by calibration and score alignment."""

from decimal import ROUND_HALF_EVEN, Decimal, localcontext

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def numeric(values, policy="raise", bounds=None):
    values = pd.Series(values, copy=False)
    if not is_numeric_dtype(values.dtype) or is_bool_dtype(values.dtype):
        raise TypeError("Expected a numeric, non-boolean column")
    array = values.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(array)
    if bounds is not None:
        valid &= (array >= bounds[0]) & (array <= bounds[1])
    if policy == "raise" and not valid.all():
        raise ValueError("Missing, non-finite or out-of-range numeric values")
    return np.where(valid, array, np.nan)


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    # exp(-abs(z)) cannot overflow, even when finite parameters multiply to inf.
    exp = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + exp), exp / (1 + exp))


def transformed(config, values):
    x = numeric(
        values, config.invalid_policy, None if config.input_space == "raw" else (0, 1)
    )
    if config.input_space == "raw":
        return x
    if config.input_probability == "non_event":
        x = 1 - x
    if config.input_space == "logit":
        x = np.clip(x, config.epsilon, 1 - config.epsilon)
        x = np.log(x) - np.log1p(-x)
    return x


def labels_and_weights(frame, target, event_label, weight, policy):
    labels = frame[target]
    if labels.isna().any() and policy == "raise":
        raise ValueError("Missing targets")
    classes = labels.dropna().unique()
    if len(classes) != 2 or event_label not in classes:
        raise ValueError("Binary targets must contain event_label and one other class")
    weights = np.ones(len(frame)) if weight is None else numeric(frame[weight])
    if (weights < 0).any():
        raise ValueError("Weights must be non-negative")
    return (
        labels.eq(event_label).to_numpy(dtype=float),
        weights,
        labels.notna().to_numpy(),
    )


def rounded(values, decimals):
    if decimals is None:
        return values.copy()
    quantum = Decimal(1).scaleb(-decimals)
    with localcontext() as context:
        context.prec = 340
        return np.array(
            [
                float(
                    Decimal(str(float(v))).quantize(quantum, rounding=ROUND_HALF_EVEN)
                )
                if np.isfinite(v)
                else np.nan
                for v in values
            ]
        )
