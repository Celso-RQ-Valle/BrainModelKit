"""Point-in-time numeric features and a complete output summary."""

import pandas as pd

from brainmodelkit.pandas import build_features

df_base = pd.DataFrame(
    {
        "account_id": ["00000000000001"] * 4,
        "observation_date": pd.to_datetime(
            [
                "2026-01-31",
                "2026-02-28",
                "2026-03-31",
                "2026-04-30",
            ]
        ),
        "revenue": [600.0, 620.0, 650.0, 700.0],
        "balance": [1000.0, None, 900.0, 1100.0],
    }
)
df_spine = pd.DataFrame(
    {
        "account_id": ["00000000000001"],
        "dt_ref": pd.to_datetime(["2026-03-31"]),
    }
)

enriched = build_features(
    df_base,
    df_spine,
    key_cols=["account_id"],
    date_col="observation_date",
    reference_date_col="dt_ref",
    value_cols=["revenue", "balance"],
    period_unit="months",
    windows=[1, 3],
    lags=[1, 2],
    aggregations=["mean", "median", "min", "max", "std", "sum", "count"],
    max_staleness_days=60,
)

# Display all generated outputs for this reference row, including lookup dates.
feature_columns = [column for column in enriched if column not in df_spine.columns]
feature_summary = (
    enriched.loc[0, feature_columns].rename_axis("feature").reset_index(name="value")
)
print(feature_summary.to_string(index=False))

# Keep only numeric features when preparing inputs for a model.
model_features = [name for name in feature_columns if not name.endswith("_date")]
