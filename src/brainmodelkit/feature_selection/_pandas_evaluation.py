"""Pandas validation and scoring consistent with the training backend.

Usage, method algorithms, parameters, and examples:
https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection.md
"""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from brainmodelkit.training._common import validate_columns

from ._pandas_quality import numeric
from ._pandas_statistics import supervised


def validate_evaluation(
    train_df: pd.DataFrame,
    oot_df: pd.DataFrame | None,
    df_scoring: pd.DataFrame | None,
    target_col: str,
    feature_cols: Sequence[str],
) -> None:
    """Validate supplied evaluation frames before any selector fitting."""
    if oot_df is None:
        return
    features = supervised(train_df, target_col, feature_cols, binary=True)
    validate_columns(features, target_col, [(oot_df, True), (df_scoring, False)])
    if not oot_df[target_col].isin([0, 1]).all():
        raise ValueError("OOT targets must be non-null binary 0/1 values")
    for frame in (oot_df, df_scoring):
        if frame is not None:
            numeric(frame, features)


def evaluate_fitted(
    model: Any,
    oot_df: pd.DataFrame,
    df_scoring: pd.DataFrame | None,
    target_col: str,
    features: list[str],
) -> Any:
    """Score the supplied model without refitting or changing its schema."""
    import numpy as np

    from brainmodelkit.metrics.pandas import _calculate_auc_gini, calculate_ks
    from brainmodelkit.training import TrainingResult

    if not hasattr(model, "predict_proba") or list(model.classes_) != [0, 1]:
        raise ValueError("OOT metrics require a fitted binary 0/1 predict_proba model")

    def score(frame):
        if frame is None:
            return None
        result = frame.copy()
        result["score"] = (
            model.predict_proba(frame[features])[:, 1]
            if len(frame)
            else np.array([], dtype=float)
        )
        return result

    predictions = score(oot_df)
    auc, gini = _calculate_auc_gini(predictions[target_col], predictions["score"])
    return TrainingResult(
        model,
        predictions,
        score(df_scoring),
        {
            "oot_ks": calculate_ks(predictions[target_col], predictions["score"]),
            "oot_auc": auc,
            "oot_gini": gini,
        },
        [],
        None,
        save_model_to="none",
    )
