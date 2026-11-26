"""Train, reload, explain, and bundle a local binary classifier."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from brainmodelkit.pandas import (
    LogisticCalibrator,
    ModelExplainer,
    PredictionPipeline,
    ScoreAligner,
)
from brainmodelkit.persistence import ArtifactSpec, load_workflow, save_workflow
from brainmodelkit.training.pandas import train_model


def run(folder):
    frame = pd.DataFrame({"income": np.arange(60) % 7, "debt": np.arange(60) % 5})
    frame["target"] = ((frame.income > frame.debt) ^ (frame.index % 4 == 0)).astype(int)
    features = ["income", "debt"]
    train, calibration, oot = frame[:30], frame[30:45], frame[45:]
    trained = train_model(
        "demo",
        "target",
        features,
        train,
        oot,
        model="logistic_regression",
        save_model_to="folder",
        model_format="joblib",
        output_dir=folder,
    )
    direct = PredictionPipeline(trained)
    held_out = direct.transform(calibration)
    calibrator = LogisticCalibrator().fit(held_out, "event_probability", "target")
    artifact = ArtifactSpec(path=trained.model_uri, format="joblib")
    workflow = PredictionPipeline(
        artifact, calibrator=calibrator, score_aligner=ScoreAligner(300, 850)
    )
    scored = workflow.transform(oot)
    np.testing.assert_allclose(scored.event_probability, direct.predict(oot))
    bundle = Path(folder) / "workflow"
    save_workflow(bundle, workflow, format="joblib")
    restored = load_workflow(bundle, format="joblib")
    explanation = ModelExplainer(
        restored,
        train,
        target="business_score",
        method="permutation",
        max_background=8,
        max_evals=9,
    )
    result = explanation.explain(scored.head(2))
    np.testing.assert_allclose(
        result.predictions, scored.business_score_unrounded.head(2)
    )
    print(result.to_frame())


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        run(directory)
