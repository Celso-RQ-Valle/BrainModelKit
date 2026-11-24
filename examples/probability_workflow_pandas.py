"""Run: python examples/probability_workflow_pandas.py --output probability_artifacts.

Install brainmodelkit[pandas,explainability,plots]. All data are synthetic.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from brainmodelkit.calibration.pandas import LogisticCalibrator, compare_calibration
from brainmodelkit.explainability.pandas import ModelExplainer, PredictionPipeline
from brainmodelkit.scoring.pandas import ScoreAligner
from brainmodelkit.training.pandas import train_model


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 2000
    data = pd.DataFrame(
        {
            "income": rng.lognormal(10, 0.5, n),
            "utilization": rng.uniform(0, 1, n),
            "late_payments": rng.poisson(1, n).astype(float),
        }
    )
    features = list(data)
    probability = 1 / (
        1
        + np.exp(
            -(
                -2
                + 2 * data.utilization
                + 0.4 * data.late_payments
                - 0.00001 * data.income
            )
        )
    )
    data["default_flag"] = rng.binomial(1, probability)
    data["customer_id"] = np.arange(n)
    data["month"] = 1
    train, development, calibration, final = (
        data.iloc[a:b].copy()
        for a, b in [(0, 1000), (1000, 1200), (1200, 1600), (1600, 2000)]
    )
    trained = train_model(
        "credit_probability",
        "default_flag",
        features,
        train,
        development,
        model=make_pipeline(StandardScaler(), LogisticRegression()),
        save_model_to="none",
    )
    # Fitting sees only held-out calibration predictions. OOF predictions could
    # replace this set; never fit calibration on in-sample training predictions.
    base = PredictionPipeline(trained, features)
    cal_predictions = base.transform(calibration)
    fitted = LogisticCalibrator(feature_order=features).fit(
        cal_predictions, "event_probability", "default_flag"
    )
    manual = LogisticCalibrator(intercept=-0.1, slope=0.9, feature_order=features)
    print(
        "Manual parameters are a transformation, not proof of calibration:",
        manual.to_dict(),
    )
    aligner = ScoreAligner(decimals=0, feature_order=features)
    workflow = PredictionPipeline(
        trained, features, calibrator=fitted, score_aligner=aligner
    )
    # Final evaluation is used only after all choices above are frozen.
    predictions = workflow.transform(final)
    reports = compare_calibration(
        predictions, "event_probability", "calibrated_event_probability", "default_flag"
    )
    print(
        {
            name: {k: v for k, v in report.items() if k != "reliability"}
            for name, report in reports.items()
        }
    )
    print(
        predictions[
            [
                "customer_id",
                "event_probability",
                "calibrated_event_probability",
                "business_score",
            ]
        ].head()
    )
    # Complementary input produces the same business score; custom scale is fixed.
    complementary = ScoreAligner(input_probability="non_event").transform(
        predictions, "calibrated_non_event_probability", output_col="complement_score"
    )
    np.testing.assert_allclose(
        complementary.complement_score, predictions.business_score_unrounded
    )
    custom = ScoreAligner(300, 850).transform(
        predictions, "calibrated_event_probability", output_col="custom_score"
    )
    print(custom[["business_score", "custom_score"]].head())
    reference = train[features].sample(20, random_state=42)
    reference.to_json(output / "reference.json", orient="table", index=False)
    dictionary = {
        "income": {
            "display_name": "Annual income",
            "units": "currency",
            "description": "Synthetic annual income",
            "group": "capacity",
        },
        "utilization": {"display_name": "Credit utilization", "group": "usage"},
    }
    for target in ("calibrated_event_probability", "business_score"):
        explainer = ModelExplainer(
            workflow,
            reference,
            target=target,
            method="permutation",
            key_cols=["customer_id", "month"],
            max_background=20,
            max_evals=35,
            reference_id="synthetic-credit-training-v1",
            feature_dictionary=dictionary,
        )
        explanation = explainer.explain(
            final[["customer_id", "month", *features]].head(3)
        )
        print(explanation.summary(0))
        explanation.to_frame().to_csv(
            output / f"{target}_explanations.csv", index=False
        )
        (output / f"{target}_metadata.json").write_text(
            json.dumps(explanation.metadata, indent=2), encoding="utf-8"
        )
    fitted.save(output / "calibration.json")
    aligner.save(output / "scoring.json")
    restored = PredictionPipeline(
        trained,
        features,
        calibrator=LogisticCalibrator.load(output / "calibration.json"),
        score_aligner=ScoreAligner.load(output / "scoring.json"),
    )
    np.testing.assert_allclose(
        restored.transform(final).business_score, predictions.business_score
    )
    print("Saved portable configurations and explanations to", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("probability_artifacts"))
    main(parser.parse_args().output)
