"""Versioned, local workflow directories; no classes imported from manifests."""

import hashlib
import json
from contextlib import suppress
from importlib import import_module, metadata
from pathlib import Path
from tempfile import TemporaryDirectory

from brainmodelkit import __version__

from ._common import local_path
from .python import _save_python_model
from .resolution import ArtifactSpec
from .spark import _save_spark_model


def _json_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported metadata value: {type(value).__name__}")


def save_workflow(
    path,
    workflow,
    *,
    format=None,
    reference=None,
    schema=None,
    feature_dictionary=None,
    background_reference=None,
    background=None,
    feature_space=None,
):
    """Save a new directory. A reference must describe the workflow's fitted model."""
    from brainmodelkit.prediction.pandas import PredictionPipeline as PandasPipeline
    from brainmodelkit.prediction.pyspark import PredictionPipeline as SparkPipeline

    if not isinstance(workflow, (PandasPipeline, SparkPipeline)):
        raise TypeError("Expected a composed PredictionPipeline")
    if getattr(workflow, "prediction_callable", None) or getattr(
        workflow, "raw_callable", None
    ):
        raise ValueError(
            "Custom callables require explicit reconstruction; cannot bundle"
        )
    if local_path(path) is None:
        raise ValueError("Use a local or mounted workflow directory")
    dest = Path(path)
    if dest.exists():
        raise FileExistsError(dest)
    backend = "pyspark" if isinstance(workflow, SparkPipeline) else "pandas"
    if reference is None and format is None:
        format = "spark" if backend == "pyspark" else "pickle"
    if reference is not None:
        if not isinstance(reference, ArtifactSpec):
            raise TypeError("reference must be an ArtifactSpec")
        if format is not None and format != reference.format:
            raise ValueError("Conflicting reference format")
        format = reference.format
    elif format not in (
        {"spark"}
        if backend == "pyspark"
        else {"pickle", "joblib", "skops", "native", "cloudpickle"}
    ):
        raise ValueError(
            "Use a supported bundle format or an explicit external reference"
        )
    if background is not None and (
        backend != "pandas" or background_reference is not None
    ):
        raise ValueError(
            "Embedded background requires pandas and no external reference"
        )
    if feature_dictionary is None:
        feature_dictionary = getattr(workflow, "feature_dictionary", {})
    features = workflow.feature_cols
    fitted_schema = getattr(workflow, "schema", None)
    if schema is not None and fitted_schema is not None and schema != fitted_schema:
        raise ValueError("Conflicting workflow and supplied schema dtypes")
    schema = schema if schema is not None else fitted_schema
    if schema is not None and list(schema) != features:
        raise ValueError("Ordered schema must match feature_cols")
    if set(feature_dictionary or {}) - set(features):
        raise ValueError("Feature dictionary includes unknown features")
    if background_reference is None and background is None:
        background_reference = getattr(workflow, "background_reference", None)
    if background_reference is not None:
        digest = background_reference.get("sha256", "")
        if (
            not background_reference.get("uri")
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in digest)
        ):
            raise ValueError(
                "Background reference requires uri and a SHA-256 hex digest"
            )
    versions = {"brainmodelkit": __version__}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "pyspark",
        "lightgbm",
        "xgboost",
        "catboost",
        "onnxruntime",
        "shap",
    ):
        with suppress(metadata.PackageNotFoundError):
            versions[name] = metadata.version(name)
    feature_space = feature_space or getattr(workflow, "feature_space", "original")
    if feature_space not in {"original", "transformed"}:
        raise ValueError("Declare original or transformed model input features")
    manifest = dict(
        version=1,
        backend=backend,
        external=reference is not None,
        path=str(reference.path) if reference else "model",
        format=format,
        flavor=reference.flavor if reference else None,
        feature_cols=features,
        schema=schema,
        feature_space=feature_space,
        adapter=type(workflow.adapter).__module__
        + "."
        + type(workflow.adapter).__qualname__
        if backend == "pandas"
        else "spark.native",
        class_labels=getattr(workflow, "classes", [0, 1]),
        event_label=workflow.event_label,
        calibration=workflow.calibrator.to_dict() if workflow.calibrator else None,
        scoring=workflow.score_aligner.to_dict() if workflow.score_aligner else None,
        probability_col=getattr(workflow, "probability_col", None),
        probability_output=getattr(workflow, "probability_output", None),
        feature_dictionary=feature_dictionary or {},
        background_reference=background_reference,
        versions=versions,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as temp:
        stage = Path(temp) / "workflow"
        stage.mkdir()
        if reference is None:
            writer = _save_spark_model if backend == "pyspark" else _save_python_model
            writer(workflow.model, stage / "model", format, save_metadata=False)
        if background is not None:
            from brainmodelkit.prediction.adapters import select_inputs

            file = stage / "background.parquet"
            select_inputs(background, features).to_parquet(file)
            manifest["embedded_background_sha256"] = hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
        (stage / "workflow.json").write_text(
            json.dumps(manifest, default=_json_scalar, allow_nan=False, indent=2),
            encoding="utf-8",
        )
        stage.rename(dest)


def load_workflow(path, *, format, reference=None, model_class=None, trusted=None):
    """The caller selects the trusted format and authorizes any external reference."""
    if local_path(path) is None:
        raise ValueError("Use a local or mounted workflow directory")
    root = Path(path)
    manifest = json.loads((root / "workflow.json").read_text(encoding="utf-8"))
    if manifest["version"] != 1 or manifest["feature_space"] not in {
        "original",
        "transformed",
    }:
        raise ValueError("Unsupported workflow manifest version or feature space")
    backend = manifest["backend"]
    if backend not in {"pandas", "pyspark"} or manifest["format"] != format:
        raise ValueError("Unexpected workflow backend or format")
    if manifest["external"]:
        if not isinstance(reference, ArtifactSpec) or (
            str(reference.path),
            reference.format,
            reference.flavor,
        ) != (manifest["path"], format, manifest["flavor"]):
            raise ValueError("Supply the matching external ArtifactSpec")
        source = reference
    else:
        location = root / "model"
        if (
            reference
            or manifest["path"] != "model"
            or location.resolve().parent != root.resolve()
        ):
            raise ValueError("Invalid internal model path")
        source = ArtifactSpec(
            path=location, format=format, model_class=model_class, trusted=trusted
        )
    options = {}
    if backend == "pyspark" and manifest["class_labels"] != [0, 1]:
        raise ValueError("Spark requires class indices [0, 1]")
    if backend == "pyspark":
        options["probability_col"] = manifest.get("probability_col")
    if backend == "pandas":
        options.update(
            class_labels=manifest["class_labels"],
            probability_output=manifest["probability_output"],
        )
    for key, module, cls in (
        ("calibration", "calibration", "LogisticCalibrator"),
        ("scoring", "scoring", "ScoreAligner"),
    ):
        if manifest[key] is not None:
            config = getattr(import_module(f"brainmodelkit.{module}.{backend}"), cls)
            options["calibrator" if key == "calibration" else "score_aligner"] = (
                config.from_dict(manifest[key])
            )
    pipeline = import_module(f"brainmodelkit.prediction.{backend}").PredictionPipeline(
        source, manifest["feature_cols"], event_label=manifest["event_label"], **options
    )
    if (
        backend == "pandas"
        and manifest["adapter"]
        != type(pipeline.adapter).__module__ + "." + type(pipeline.adapter).__qualname__
    ):
        raise ValueError("Register the original prediction adapter before loading")
    if pipeline.schema and manifest["schema"] and pipeline.schema != manifest["schema"]:
        raise ValueError("Conflicting artifact and workflow dtypes")
    pipeline.feature_space = manifest["feature_space"]
    pipeline.schema = (
        manifest["schema"] if manifest["schema"] is not None else pipeline.schema
    )
    if pipeline.schema is not None and list(pipeline.schema) != pipeline.feature_cols:
        raise ValueError("Conflicting workflow schema")
    pipeline.feature_dictionary = manifest["feature_dictionary"]
    if set(pipeline.feature_dictionary) - set(pipeline.feature_cols):
        raise ValueError("Workflow feature dictionary includes unknown features")
    pipeline.background_reference = manifest["background_reference"]
    if "embedded_background_sha256" in manifest:
        import pandas as pd

        file = root / "background.parquet"
        if (
            file.resolve().parent != root.resolve()
            or hashlib.sha256(file.read_bytes()).hexdigest()
            != manifest["embedded_background_sha256"]
        ):
            raise ValueError("Embedded background integrity check failed")
        pipeline.background = pd.read_parquet(file)
    return pipeline
