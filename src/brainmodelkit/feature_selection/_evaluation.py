"""Shared selection-to-training orchestration, with backend-injected trainers.

Usage, method algorithms, parameters, and examples:
https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection.md
"""

from collections.abc import Callable, Sequence
from typing import Any

from ._selection import SelectionResult, resolve_frame


def evaluation_frame(oot_df: Any, df_oot: Any, df_scoring: Any) -> Any:
    oot_df = resolve_frame(oot_df, df_oot, "oot_df", "df_oot", required=False)
    if df_scoring is not None and oot_df is None:
        raise ValueError("df_scoring requires oot_df for final model evaluation")
    return oot_df


def finalize_selection(
    result: SelectionResult,
    trainer: Callable,
    *,
    train_df: Any,
    oot_df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    model: Any,
    df_scoring: Any = None,
    run_name: str = "feature_selection",
    **backend_options: Any,
) -> SelectionResult:
    """Refit selected features on training data; OOT only evaluates the final fit."""
    result.metadata.update(
        run_name=run_name,
        target_col=target_col,
        input_features=list(feature_cols),
        evaluation_status="not_requested",
    )
    if oot_df is None:
        return result
    result.selection_model = result.model
    if not result.selected_features:
        result.model = None
        result.metadata["evaluation_status"] = "no_selected_features"
        return result
    result.training_result = trainer(
        run_name=run_name,
        target_col=target_col,
        feature_cols=result.selected_features,
        train_df=train_df,
        oot_df=oot_df,
        model=model,
        df_scoring=df_scoring,
        save_model_to="none",
        **backend_options,
    )
    result.model = result.training_result.model
    result.metadata.update(
        evaluation_status="evaluated", model_feature_cols=result.selected_features
    )
    return result
