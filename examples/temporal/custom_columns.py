"""Run: python examples/temporal/custom_columns.py (requires pandas extra)."""

import pandas as pd

from brainmodelkit.pandas import build_features

df_base = pd.DataFrame(
    {
        "customer_id": ["c1", "c1", "c1", "c1"],
        "account_id": ["a1", "a1", "a1", "a2"],
        "observation_date": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-03-01"],
        "used_balance": [10.0, None, 20.0, 90.0],
        "overdue_balance": [1.0, 2.0, None, 5.0],
        "credit_limit": [100.0, 0.0, 100.0, 200.0],
    }
)
df_spine = pd.DataFrame(
    {
        "customer_id": ["c1", "c1"],
        "account_id": ["a1", "a2"],
        "dt_ref": ["2026-03-01", "2026-03-01"],
    }
)
create_cols = {
    "total_balance": {
        "op": "sum",
        "args": [{"col": "used_balance"}, {"col": "overdue_balance"}],
    },
    "total_utilization": {
        "op": "divide",
        "args": [
            {
                "op": "sum",
                "args": [{"col": "used_balance"}, {"col": "overdue_balance"}],
            },
            {"col": "credit_limit"},
        ],
    },
}
# These options do not duplicate any explicitly passed keyword arguments.
feature_options = dict(windows=[3], lags=[1], period_unit="months")
enriched = build_features(
    df_base=df_base,
    df_spine=df_spine,
    key_cols=["customer_id", "account_id"],
    value_cols=["used_balance", "credit_limit", "total_balance", "total_utilization"],
    create_cols=create_cols,
    nullable=False,
    relative_historico=True,
    **feature_options,
)
default = build_features(
    df_base=df_base,
    df_spine=df_spine,
    key_cols=["customer_id", "account_id"],
    value_cols=["used_balance", "credit_limit"],
    create_cols=None,
    nullable=True,
    **feature_options,
)
assert enriched.loc[0, "used_balance_mean_3_months"] == 10
assert default.loc[0, "used_balance_mean_3_months"] == 15
print(
    enriched[
        [
            "customer_id",
            "account_id",
            "total_balance_mean_3_months",
            "total_utilization_mean_3_months",
        ]
    ].to_string(index=False)
)
