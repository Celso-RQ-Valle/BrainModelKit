"""Backend-independent recursive feature elimination."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from brainmodelkit.training import TrainingResult
from brainmodelkit.training._common import validate_columns


@dataclass
class RFEResult:
    """Selected columns, final fitted training result and iteration reports."""

    selected_features: list[str]
    training_result: TrainingResult
    history: list[dict]
    output_dir: Path


def run_rfe(
    trainer,
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
    **backend_options,
):
    for name, value in (("step", step), ("n_feat_final", n_feat_final)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if feature_cols is None:
        feature_cols = [col for col in train_df.columns if col.startswith("feat_")]
    features = validate_columns(
        feature_cols,
        target_col,
        [(train_df, True), (oot_df, True), (df_scoring, False)],
    )
    if n_feat_final > len(features):
        raise ValueError("n_feat_final cannot exceed the initial feature count")
    folder = Path(output_dir) / uuid4().hex
    history = []
    while True:
        result = trainer(
            run_name=f"{run_name}_{len(features)}_feats",
            target_col=target_col,
            feature_cols=features.copy(),
            train_df=train_df,
            oot_df=oot_df,
            model=model,
            model_params=model_params,
            output_dir=folder,
            df_scoring=df_scoring if len(features) == n_feat_final else None,
            run_as="mlflow" if mlflow_logging else "local",
            signature=signature,
            **backend_options,
        )
        importance = result.feature_importance
        if (
            len(importance) != len(features)
            or {row["feature"] for row in importance} != set(features)
            or any(not math.isfinite(row["importance"]) for row in importance)
        ):
            raise ValueError(
                "RFE requires one finite native importance or coefficient per feature; "
                "this estimator does not provide compatible feature importance"
            )
        # Stable ties follow the original feature order. Negative coefficients
        # are ranked by magnitude, so strong negative effects are retained.
        values = {row["feature"]: abs(row["importance"]) for row in importance}
        count = min(step, len(features) - n_feat_final)
        removed = sorted(features, key=values.__getitem__)[:count]
        history.append(
            {
                "iteration": len(history) + 1,
                "n_features": len(features),
                "features": features.copy(),
                "removed_features": removed,
                "metrics": {
                    key: value if math.isfinite(value) else None
                    for key, value in result.metrics.items()
                },
                "report_dir": str(result.output_dir) if result.output_dir else None,
                "run_id": result.run_id,
            }
        )
        folder.mkdir(parents=True, exist_ok=True)
        # Each completed iteration remains reviewable if a later fit fails.
        (folder / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=False), encoding="utf-8"
        )
        if not count:
            break
        features = [feature for feature in features if feature not in removed]
    (folder / "selected_features.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8"
    )
    return RFEResult(features, result, history, folder)
