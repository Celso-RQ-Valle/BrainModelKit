"""Shared training results and local run reports."""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class TrainingResult:
    """Fitted model, scored frames, OOT metrics and local report location."""

    model: Any
    oot_predictions: Any
    scoring_predictions: Any
    metrics: dict[str, float]
    feature_importance: list[dict]
    output_dir: Path
    run_id: str | None = None


def validate_columns(feature_cols, target_col, frames):
    if isinstance(feature_cols, str) or not feature_cols:
        raise ValueError("feature_cols must be a non-empty sequence of column names")
    features = list(feature_cols)
    if len(set(features)) != len(features) or target_col in features:
        raise ValueError("Features must be unique and must not include the target")
    for frame, labeled in frames:
        if frame is None:
            continue
        required = features + ([target_col] if labeled else [])
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        if "score" in frame.columns:
            raise ValueError("Input column 'score' is reserved for model output")
    return features


def finish(
    result,
    run_name,
    backend,
    features,
    target,
    params,
    output_dir,
    mlflow_logging,
    signature,
    train_df,
):
    folder = Path(output_dir) / uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    result.output_dir = folder
    report = {
        "run_name": run_name,
        "backend": backend,
        "model_class": type(result.model).__name__,
        "estimator_class": type(
            result.model.stages[-1] if backend == "spark" else result.model
        ).__name__,
        "feature_cols": features,
        "target_col": target,
        "parameters": params,
        "metrics": {
            k: v if math.isfinite(v) else None for k, v in result.metrics.items()
        },
    }
    (folder / "model_info.json").write_text(
        json.dumps(report, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )
    with (folder / "feature_importance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["feature", "importance"])
        writer.writeheader()
        writer.writerows(result.feature_importance)
    if mlflow_logging:
        import mlflow
        from mlflow.models import infer_signature

        flavor = __import__(f"mlflow.{backend}", fromlist=["log_model"])
        with mlflow.start_run(
            run_name=run_name, nested=mlflow.active_run() is not None
        ):
            result.run_id = mlflow.active_run().info.run_id
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            mlflow.log_metrics(
                {k: v for k, v in result.metrics.items() if math.isfinite(v)}
            )
            model_signature = None
            if signature:
                if backend == "sklearn":
                    sample = train_df[features].head(5)
                    model_signature = infer_signature(
                        sample, result.model.predict(sample)
                    )
                else:
                    model_signature = infer_signature(
                        train_df.select(*features),
                        result.oot_predictions.select("prediction"),
                    )
            flavor.log_model(
                result.model, artifact_path="model", signature=model_signature
            )
            mlflow.log_artifacts(str(folder), artifact_path="training_report")
        (folder / "mlflow_run_id.txt").write_text(result.run_id, encoding="utf-8")
    return result


def importance_records(features, values):
    if values is None:
        return []
    return sorted(
        [
            {"feature": name, "importance": float(value)}
            for name, value in zip(features, values, strict=True)
        ],
        key=lambda row: abs(row["importance"]),
        reverse=True,
    )
