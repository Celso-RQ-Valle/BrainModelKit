"""Pandas-native feature selection and backward-compatible OOT RFE."""

from brainmodelkit.training.pandas import train_model

from ._common import RFEResult, run_rfe
from ._pandas_model import (
    feature_importance_selection,
    l1_selection,
    permutation_importance_selection,
)
from ._pandas_quality import cardinality, completeness, variance_filter
from ._pandas_statistics import (
    anova,
    chi_square,
    correlation_filter,
    information_value,
    mutual_information,
)
from ._pandas_wrappers import boruta, rfecv, sequential_selection, stability_selection
from ._selection import resolve_frame

__all__ = [
    "anova",
    "boruta",
    "cardinality",
    "chi_square",
    "completeness",
    "correlation_filter",
    "feature_importance_selection",
    "information_value",
    "l1_selection",
    "mutual_information",
    "permutation_importance_selection",
    "rfe",
    "rfecv",
    "sequential_selection",
    "stability_selection",
    "variance_filter",
]


def rfe(
    run_name,
    target_col,
    train_df,
    oot_df=None,
    feature_cols=None,
    model="logistic_regression",
    *,
    df_oot=None,
    step=1,
    n_feat_final=1,
    model_params=None,
    output_dir="feature_selection_runs",
    df_scoring=None,
    mlflow_logging=False,
    signature=False,
) -> RFEResult:
    """Refit after removing the least important features, ending at n_feat_final.

    Supports logistic_regression, random_forest, gradient_boosting, lightgbm,
    or cloneable classifiers with predict_proba and native feature_importances_
    or coef_. Importance is ranked by absolute magnitude; ties remove earlier
    input columns first. Retained feature order is preserved. Scale features
    before using coefficient magnitudes when their units differ.
    Omitted feature_cols selects columns starting with ``feat_``.
    OOT metrics are diagnostic only and do not choose the subset.
    Each fitted subset gets a training report under a unique output directory.
    Only the final fit scores df_scoring. Returns selected_features, the final
    training_result, history and output_dir. Inputs and estimators are not mutated.
    Custom models must expose one finite importance per input feature.
    """
    oot_df = resolve_frame(oot_df, df_oot, "oot_df", "df_oot")
    return run_rfe(
        train_model,
        run_name,
        target_col,
        feature_cols,
        train_df,
        oot_df,
        model,
        step,
        n_feat_final,
        model_params,
        output_dir,
        df_scoring,
        mlflow_logging,
        signature,
    )
