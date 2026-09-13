"""Native Spark validation for optional final-model OOT evaluation."""

from collections.abc import Sequence

from pyspark.sql import DataFrame

from brainmodelkit.training._common import validate_columns

from ._spark_quality import col, numeric, present
from ._spark_statistics import supervised


def validate_evaluation(
    train_df: DataFrame,
    oot_df: DataFrame | None,
    df_scoring: DataFrame | None,
    target_col: str,
    feature_cols: Sequence[str],
) -> None:
    if oot_df is None:
        return
    features = supervised(train_df, target_col, feature_cols)
    validate_columns(features, target_col, [(oot_df, True), (df_scoring, False)])
    if (
        oot_df.filter(~present(oot_df, target_col) | ~col(target_col).isin(0, 1))
        .limit(1)
        .count()
    ):
        raise ValueError("OOT targets must be non-null binary 0/1 values")
    for frame in (oot_df, df_scoring):
        if frame is not None:
            numeric(frame, features)
