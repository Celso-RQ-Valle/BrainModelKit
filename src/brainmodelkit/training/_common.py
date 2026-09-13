"""Shared training results, destination validation and persistence orchestration."""

import csv
import json
import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from platform import python_version
from tempfile import TemporaryDirectory
from typing import Any, Literal
from uuid import uuid4

from brainmodelkit import __version__
from brainmodelkit.persistence._common import (
    PYTHON_FORMATS,
    optional_import,
    serializable_parameters,
    validate_format,
)
from brainmodelkit.persistence.mlflow import _log_mlflow_model
from brainmodelkit.persistence.python import _save_python_model
from brainmodelkit.persistence.spark import _save_spark_model


class _DefaultDestination(str):
    """Distinguish omitted destination from an explicitly supplied folder."""


DEFAULT_DESTINATION = _DefaultDestination("folder")


@dataclass
class TrainingResult:
    """Fitted model, scored frames, OOT metrics and persistence details."""

    model: Any
    oot_predictions: Any
    scoring_predictions: Any
    metrics: dict[str, float]
    feature_importance: list[dict]
    output_dir: Path | None
    run_id: str | None = None
    save_model_to: Literal["folder", "mlflow", "none"] = "folder"
    model_uri: str | None = None
    model_format: str | None = None

    @property
    def run_as(self):
        """Deprecated destination spelling, retaining the historical values."""
        warnings.warn(
            "'run_as' is deprecated; use 'save_model_to' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return "local" if self.save_model_to == "folder" else self.save_model_to


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


def resolve_execution(
    backend,
    save_model_to,
    model_format,
    mlflow_logging,
    save_path,
    save_format,
    signature,
    *,
    run_as=None,
    runs_as=None,
):
    """Normalize deprecated switches and validate before fitting."""
    if save_model_to not in ("folder", "mlflow", "none"):
        raise ValueError("Invalid save_model_to. Available: folder, mlflow, none")
    destinations = [] if save_model_to is DEFAULT_DESTINATION else [save_model_to]
    for name, value in (("run_as", run_as), ("runs_as", runs_as)):
        if value is not None:
            warnings.warn(
                f"'{name}' is deprecated and will be removed in a future "
                "BrainModelKit release. Use 'save_model_to' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            validate_format(value, ("local", "mlflow", "none"))
            destinations.append("folder" if value == "local" else value)
    for name, value in (
        ("mlflow_logging", mlflow_logging),
        ("save_path", save_path),
        ("save_format", save_format),
    ):
        if value is not None:
            warnings.warn(
                f"{name} is deprecated; use save_model_to, "
                "model_format and output_dir.",
                DeprecationWarning,
                stacklevel=3,
            )
    if mlflow_logging:
        destinations.append("mlflow")
    if len(set(destinations)) > 1:
        raise ValueError(
            "save_model_to conflicts with legacy aliases "
            "(run_as, runs_as or mlflow_logging); use save_model_to only"
        )
    destination = destinations[0] if destinations else "folder"
    if save_format == "mlflow":
        raise ValueError(
            "save_format='mlflow' was replaced by save_model_to='mlflow'; "
            "omit save_path"
        )
    if save_format is not None:
        if model_format is not None and model_format != save_format:
            raise ValueError("save_format conflicts with model_format")
        model_format = save_format
    available = (
        ("spark",)
        if backend == "spark"
        else tuple(value for value in PYTHON_FORMATS if value != "mlflow")
    )
    if model_format is not None:
        validate_format(model_format, available)
    if destination != "folder" and (model_format is not None or save_path is not None):
        raise ValueError("model_format and save_path require save_model_to='folder'")
    if signature and destination != "mlflow":
        raise ValueError("signature=True requires save_model_to='mlflow'")
    if destination == "mlflow":
        optional_import("mlflow", "mlflow")
    return destination, (
        model_format or available[0]
    ) if destination == "folder" else None


def _write_reports(folder, result, report, metadata):
    for name, content in (
        ("model_info.json", report),
        ("metrics.json", result.metrics),
        ("metadata.json", metadata),
    ):
        # Undefined metrics are JSON null; other parameters remain readable.
        if name == "metrics.json":
            content = {k: v if math.isfinite(v) else None for k, v in content.items()}
        (folder / name).write_text(
            json.dumps(content, indent=2, default=str, allow_nan=False),
            encoding="utf-8",
        )
    with (folder / "feature_importance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["feature", "importance"])
        writer.writeheader()
        writer.writerows(result.feature_importance)


def finalize_run(
    result,
    run_name,
    backend,
    features,
    target,
    params,
    output_dir,
    save_model_to,
    model_format,
    signature,
    train_df,
    *,
    save_path=None,
    save_metadata=True,
    overwrite=False,
):
    """Persist a complete run through one destination."""
    result.save_model_to = save_model_to
    result.model_format = model_format
    result.output_dir = None
    if save_model_to == "none":
        return result
    result.run_id = uuid4().hex
    metadata = {
        "brainmodelkit_version": __version__,
        "python_version": python_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "save_model_to": save_model_to,
        "model_format": model_format,
        "framework": type(result.model).__module__.split(".")[0],
        "model_class": type(result.model).__name__,
        "backend": backend,
    }
    report = {
        "run_name": run_name,
        "run_id": result.run_id,
        "backend": backend,
        "model_class": type(result.model).__name__,
        "estimator_class": type(
            result.model.stages[-1] if backend == "spark" else result.model
        ).__name__,
        "feature_cols": features,
        "target_col": target,
        "parameters": serializable_parameters(params),
    }
    if save_model_to == "folder":
        safe_name = (
            re.sub(r"[^A-Za-z0-9_-]+", "_", str(run_name)).strip("_")[:80] or "run"
        )
        folder = Path(output_dir) / f"{safe_name}_{result.run_id}"
        folder.mkdir(parents=True, exist_ok=False)
        result.output_dir = folder
        suffixes = {
            "pickle": ".pkl",
            "joblib": ".joblib",
            "cloudpickle": ".pkl",
            "skops": ".skops",
            "onnx": ".onnx",
            "native": "",
            "spark": "",
        }
        path = (
            save_path
            if save_path is not None
            else folder / ("model" + suffixes[model_format])
        )
        options = dict(
            save_metadata=save_metadata if save_path is not None else False,
            target_col=target,
            feature_cols=features,
            parameters=params,
        )
        if backend == "spark":
            _save_spark_model(
                result.model, path, model_format, overwrite=overwrite, **options
            )
        else:
            _save_python_model(
                result.model,
                path,
                model_format,
                input_example=train_df[features].head(1)
                if model_format == "onnx"
                else None,
                **options,
            )
        result.model_uri = str(path)
        _write_reports(folder, result, report, metadata)
        return result

    mlflow = optional_import("mlflow", "mlflow")
    with mlflow.start_run(
        run_name=run_name, nested=mlflow.active_run() is not None
    ) as run:
        result.run_id = run.info.run_id
        report["run_id"] = result.run_id
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.log_metrics(
            {k: v for k, v in result.metrics.items() if math.isfinite(v)}
        )
        mlflow.set_tags(
            {
                **{k: str(v) for k, v in metadata.items()},
                "target_col": target,
                "feature_cols": json.dumps(features),
            }
        )
        model_signature = None
        if signature:
            infer_signature = optional_import("mlflow.models", "mlflow").infer_signature
            if backend == "sklearn":
                sample = train_df[features].head(5)
                model_signature = infer_signature(sample, result.model.predict(sample))
            else:
                model_signature = infer_signature(
                    train_df.select(*features),
                    result.oot_predictions.select("prediction"),
                )
        _log_mlflow_model(result.model, backend, model_signature)
        result.model_uri = f"runs:/{result.run_id}/model"
        with TemporaryDirectory(prefix="brainmodelkit-") as temporary:
            _write_reports(Path(temporary), result, report, metadata)
            mlflow.log_artifacts(temporary, artifact_path="training_report")
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
