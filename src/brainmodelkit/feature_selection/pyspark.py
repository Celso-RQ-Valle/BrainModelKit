"""Recursive feature elimination for Spark binary classifiers."""

from brainmodelkit.training.pyspark import train_model

from ._common import RFEResult, run_rfe


def rfe(
    run_name,
    target_col,
    train_df,
    oot_df,
    feature_cols=None,
    model="logistic_regression",
    *,
    step=1,
    n_feat_final=1,
    model_params=None,
    output_dir="feature_selection_runs",
    df_scoring=None,
    mlflow_logging=False,
    signature=False,
    n_tiles=10,
) -> RFEResult:
    """Refit after removing the least important features, ending at n_feat_final.

    Supports the training module's logistic_regression, random_forest,
    gradient_boosting and lightgbm models, or compatible custom estimators.
    Importance is native tree importance or absolute linear coefficient.
    Ties remove earlier input columns first; retained feature order is preserved.
    Omitted feature_cols selects columns starting with ``feat_``.
    OOT metrics are diagnostic only and do not choose the subset. Spark metrics
    use n_tiles. Frames remain distributed and caller caches are not changed.
    Reports for every fitted subset are written under a unique output directory;
    only the final fit scores df_scoring. Returns the final training result,
    selected_features, history and output_dir. SynapseML must be configured for
    LightGBM. Custom models must expose one finite importance per input feature.
    """
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
        n_tiles=n_tiles,
    )
