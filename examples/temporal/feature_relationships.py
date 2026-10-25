"""Integrated temporal families using the README simulated credit export.

Run: python examples/temporal/feature_relationships.py simulated_credit.csv
Generate the CSV with the Spark prepare-data snippet in README.md first.
"""

import argparse

import pandas as pd

from brainmodelkit.pandas import build_features, simulate_bureau_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "credit_csv", help="Simulated credit CSV exported by the README"
    )
    args = parser.parse_args()
    df_spine = pd.read_csv(
        args.credit_csv,
        dtype={"account_id": "string"},
        parse_dates=["reference_date"],
    )
    df_base = simulate_bureau_history(
        df_spine[["account_id"]],
        entity_cols="account_id",
        initial_date="2025-10-01",
        final_date="2026-06-30",
        seed=42,
    )
    # Reproducible illustrative balances and changing limits, on the same history rows.
    df_base = df_base.assign(
        used_balance=df_base["bureau_score"] * 0.2,
        credit_limit=100.0
        + pd.to_datetime(df_base["observation_date"]).dt.month * 20.0,
    )
    enriched = build_features(
        df_base,
        df_spine,
        key_cols="account_id",
        reference_date_col="reference_date",
        value_cols=["used_balance", "credit_limit"],
        windows=[1, 3, 6],
        lags=[1, 2, 3],
        period_unit="months",
        aggregations=["mean", "median", "min", "max", "std", "sum", "count"],
        ref_col="credit_limit",
        relative_historico=True,
    )
    generated_features = [c for c in enriched.columns if c not in df_spine.columns]
    credit_features = [f"feature_{i:02d}" for i in range(1, 21)]
    candidate_features = credit_features + generated_features
    prefix = "used_balance__over__credit_limit__"
    print(
        enriched[
            [
                "account_id",
                "reference_date",
                prefix + "current",
                prefix + "mean_3_months",
                prefix + "current_vs_lag_1_months_diff",
                prefix + "current_vs_lag_1_months_diff_pp",
                prefix + "current_vs_lag_1_months_pct_change",
            ]
        ]
        .head()
        .to_string(index=False)
    )
    print(
        f"Generated {len(generated_features)} features; "
        f"{len(candidate_features)} total model candidates."
    )


if __name__ == "__main__":
    main()
