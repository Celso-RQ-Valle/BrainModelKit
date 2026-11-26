"""Resolve explicitly selected artifacts and recover training metadata."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._common import local_path


@dataclass(frozen=True, kw_only=True)
class ArtifactSpec:
    path: str | Path
    format: str
    model_class: type | None = None
    trusted: list[str] | None = None
    flavor: str | None = None


@dataclass
class ResolvedModel:
    model: Any
    feature_cols: list | None
    class_labels: list | None
    schema: dict | None = None


def reconcile(name, *values):
    present = [list(v) for v in values if v is not None]
    if present and any(v != present[0] for v in present):
        raise ValueError(f"Conflicting {name} metadata")
    return present[0] if present else None


def resolve_model(source, *, feature_cols=None, class_labels=None):
    from brainmodelkit.training import TrainingResult

    from . import load_model

    records = []
    if isinstance(source, ArtifactSpec):
        path = local_path(source.path)
        if path is not None:
            files = [Path(str(path) + ".metadata.json")]
            if (path.parent / "model_info.json").is_file() and path.name in {
                "model",
                "model.pkl",
                "model.joblib",
                "model.skops",
                "model.onnx",
            }:
                files += [
                    path.parent / "metadata.json",
                    path.parent / "model_info.json",
                ]
            for file in files:
                if file.is_file():
                    record = json.loads(file.read_text(encoding="utf-8"))
                    declared = record.get("model_format") or record.get("save_format")
                    if declared and declared != source.format:
                        raise ValueError("Conflicting artifact format metadata")
                    records.append(record)
        source = load_model(
            source.path,
            source.format,
            model_class=source.model_class,
            trusted=source.trusted,
            flavor=source.flavor,
        )
    elif isinstance(source, (str, Path)):
        raise TypeError("Use ArtifactSpec(path=..., format=...) for saved models")
    if isinstance(source, TrainingResult):
        records.append(
            {
                "feature_cols": getattr(source, "feature_cols", None),
                "schema": getattr(source, "schema", None),
            }
        )
        source = source.model
    from brainmodelkit.prediction import Workflow

    if isinstance(source, Workflow):
        records.append({"schema": getattr(source, "schema", None)})
        feature_cols = reconcile("features", feature_cols, source.feature_cols)
        class_labels = reconcile(
            "classes", class_labels, getattr(source, "classes", None)
        )
        source = source.model
    names = getattr(source, "feature_names_in_", None)
    framework = type(source).__module__.split(".")[0]
    if names is None and framework == "xgboost":
        names = getattr(source, "feature_names", None)
    if names is None and framework == "lightgbm" and hasattr(source, "feature_name"):
        native_names = source.feature_name()
        if native_names != [f"Column_{i}" for i in range(len(native_names))]:
            names = native_names
    if names is None and framework == "catboost":
        native_names = getattr(source, "feature_names_", None)
        if native_names and native_names != [str(i) for i in range(len(native_names))]:
            names = native_names
    schemas = [r["schema"] for r in records if r.get("schema") is not None]
    schema = schemas[0] if schemas else None
    if any(candidate != schema for candidate in schemas):
        raise ValueError("Conflicting artifact schema dtypes")
    features = reconcile(
        "features", feature_cols, *(r.get("feature_cols") for r in records), names
    )
    if schema is not None:
        if features is None:
            features = list(schema)
        if set(schema) != set(features):
            raise ValueError("Conflicting schema and model features")
        schema = {name: schema[name] for name in features}
    return ResolvedModel(
        source,
        features,
        reconcile(
            "classes",
            class_labels,
            *(r.get("class_labels") for r in records),
            getattr(source, "classes_", None),
        ),
        schema,
    )
