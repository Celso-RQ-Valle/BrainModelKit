"""Show all aggregates and relationships using the README simulated credit export.

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
    credit_features = [f"feature_{i:02d}" for i in range(1, 21)]
    df_base = simulate_bureau_history(
        df_spine[["account_id"]],
        entity_cols="account_id",
        initial_date="2025-10-01",
        final_date="2026-06-30",
        seed=42,
    )

    enriched = build_features(
        df_base,
        df_spine,
        key_cols="account_id",
        value_cols=["bureau_score"],
        reference_date_col="reference_date",
        windows=[1, 3, 6],
        lags=[1, 2, 3],
        period_unit="months",
        aggregations=["mean", "median", "min", "max", "std", "sum", "count"],
    )
    history_features = [c for c in enriched.columns if c not in df_spine.columns]
    candidate_features = credit_features + history_features

    current = enriched["bureau_score__current"]
    lag = enriched["bureau_score__lag_1_months"]
    mean = enriched["bureau_score__mean_3_months"]
    std = enriched["bureau_score__std_3_months"]

    def safe_ratio(numerator, denominator):
        return numerator / denominator.where(denominator.ne(0))

    relationships = {
        "bureau_score__change_1_month": current - lag,
        "bureau_score__ratio_1_month": safe_ratio(current, lag),
        "bureau_score__relative_change_1_month": safe_ratio(current - lag, lag),
        "bureau_score__current_minus_median_3_months": (
            current - enriched["bureau_score__median_3_months"]
        ),
        "bureau_score__range_3_months": (
            enriched["bureau_score__max_3_months"]
            - enriched["bureau_score__min_3_months"]
        ),
        "bureau_score__standardized_deviation_3_months": safe_ratio(
            current - mean, std
        ),
        "bureau_score__coefficient_of_variation_3_months": safe_ratio(std, mean.abs()),
        "bureau_score__mean_ratio_1_to_6_months": safe_ratio(
            enriched["bureau_score__mean_1_months"],
            enriched["bureau_score__mean_6_months"],
        ),
    }
    enriched = enriched.assign(**relationships)
    candidate_features = credit_features + history_features + list(relationships)
    print(enriched[["account_id", "reference_date", *relationships]].head())

    print(
        f"Generated {len(history_features)} history features "
        f"and {len(relationships)} relationships; "
        f"{len(candidate_features)} total model candidates."
    )


if __name__ == "__main__":
    main()
