"""Run: python examples/journey/pandas_workflow.py [--save-dir PATH]."""

import argparse

import pandas as pd
from sample_data import make_rows

from brainmodelkit.cube_analysis.pandas import calculate_metrics, calculate_ntile
from brainmodelkit.feature_selection.pandas import variance_filter
from brainmodelkit.metrics.pandas import auc_gini, ks, risk_table
from brainmodelkit.pandas import build_features
from brainmodelkit.persistence import load_model
from brainmodelkit.training.pandas import train_model


def main(save_dir=None):
    base_rows, spine_rows = make_rows()
    df_base = pd.DataFrame(
        base_rows, columns=["account_id", "observation_date", "revenue", "balance"]
    )
    df_spine = pd.DataFrame(
        spine_rows, columns=["account_id", "reference_date", "segment", "target"]
    )
    enriched = build_features(
        df_base,
        df_spine,
        key_cols="account_id",
        value_cols=["revenue", "balance"],
        reference_date_col="reference_date",
        windows=[3],
        lags=[1],
        aggregations=["mean", "sum", "count"],
    )
    candidate_features = [c for c in enriched if c not in df_spine.columns]
    # Fixed demonstration policy: no statistics are learned from validation/OOT.
    enriched[candidate_features] = enriched[candidate_features].fillna(0)
    ref = pd.to_datetime(enriched.reference_date)
    train_df = enriched.loc[ref < "2026-05-01"].copy()
    validation_df = enriched.loc[(ref >= "2026-05-01") & (ref < "2026-06-01")].copy()
    oot_df = enriched.loc[ref >= "2026-06-01"].copy()
    selection = variance_filter(train_df, feature_cols=candidate_features)
    selected_features = selection.selected_features
    assert selected_features
    trials = []
    for c in (0.1, 1.0):
        baseline = train_model(
            "development",
            "target",
            selected_features,
            train_df,
            validation_df,
            model="logistic_regression",
            model_params={"C": c, "max_iter": 1000},
            save_model_to="none",
        )
        trials.append((baseline.metrics["oot_auc"], c))
    # Here oot_df in the training API received development validation, not final OOT.
    best_c = max(trials, key=lambda trial: trial[0])[1]
    development_df = pd.concat([train_df, validation_df], ignore_index=True)
    result = train_model(
        "final_pandas",
        "target",
        selected_features,
        development_df,
        oot_df,
        model="logistic_regression",
        model_params={"C": best_c, "max_iter": 1000},
        save_model_to="folder" if save_dir else "none",
        output_dir=save_dir or "training_runs",
    )
    predictions = result.oot_predictions
    print(result.metrics)
    print(ks(dataframe=predictions))
    print(auc_gini(df=predictions))
    print(risk_table(df=predictions, n_tiles=4))
    print(calculate_metrics(df=predictions, group_columns=["segment"]))
    print(calculate_ntile(df=predictions, group_columns=["segment"], n_tiles=4))
    new_data = oot_df.drop(columns="target")  # Schema-only scoring demonstration.
    fitted = load_model(result.model_uri, format="pickle") if save_dir else result.model
    positive = list(fitted.classes_).index(1)
    scores = fitted.predict_proba(new_data[selected_features])[:, positive]
    assert len(scores) == len(oot_df)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert len(train_df) == len(validation_df) == len(oot_df) == 40
    print("Pandas journey passed; model URI:", result.model_uri)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir")
    main(parser.parse_args().save_dir)
