"""Native distributed Spark feature selection and backward-compatible OOT RFE.

Usage, method algorithms, parameters, and examples:
https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection.md
"""

from brainmodelkit.training.pyspark import train_model

from ._common import RFEResult, run_rfe
from ._selection import resolve_frame
from ._spark_iv import information_value
from ._spark_model import feature_importance_selection, l1_selection
from ._spark_quality import cardinality, completeness, variance_filter
from ._spark_robustness import boruta, stability_selection
from ._spark_search import permutation_importance_selection, rfecv, sequential_selection
from ._spark_statistics import chi_square, correlation_filter
from ._spark_univariate import anova, mutual_information

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

    Usage
    -----
    Import `brainmodelkit.feature_selection.pyspark` as `fs`, then call:

        result = fs.rfe(
            "demo", "target", train_df, oot_df, ["income", "age"], n_feat_final=1
        )
        print(result.selected_features)

    Parameters, return fields, algorithm, assumptions, and complete examples:
    https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/rfe.md
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
        n_tiles=n_tiles,
    )
