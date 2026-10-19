"""Point-in-time revenue features. Run with the pandas extra installed."""

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
    value_cols=["revenue"],
    period_unit="months",
    windows=[3],
    lags=[1],
    aggregations=["mean", "min", "max", "std", "count"],
)

feature_columns = [
    column
    for column in enriched.columns
    if column not in df_spine.columns and not column.endswith("_date")
]
print(enriched[["account_id", "dt_ref", *feature_columns]])
