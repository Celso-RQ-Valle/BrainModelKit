"""Small dependency, validation and metadata helpers."""

import importlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from platform import python_version

from brainmodelkit import __version__

PYTHON_FORMATS = (
    "pickle",
    "joblib",
    "cloudpickle",
    "skops",
    "native",
    "onnx",
    "mlflow",
)
SPARK_FORMATS = ("spark", "mlflow")


def validate_format(format, available):
    if format not in available:
        raise ValueError(
            f"Unknown persistence format {format!r}. Available: {', '.join(available)}"
        )


def optional_import(module, extra):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module} persistence requires optional dependencies. "
            f"Install BrainModelKit with: pip install 'brainmodelkit[{extra}]'"
        ) from exc


def local_path(path):
    # Leave filesystem URIs (including file:) to the backend that owns them.
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", str(path)) and not re.match(
        r"^[a-zA-Z]:[\\/]", str(path)
    ):
        return None
    return Path(path)


def serializable_parameters(parameters):
    """Keep parameters that can be represented faithfully in standard JSON."""
    params = {}
    for key, value in (parameters or {}).items():
        try:
            json.dumps({key: value}, allow_nan=False)
        except (TypeError, ValueError, OverflowError):
            continue
        params[key] = value
    return params


def write_metadata(
    model, path, format, *, target_col=None, feature_cols=None, parameters=None
):
    local = local_path(path)
    if local is None:
        return
    metadata = {
        "model_class": type(model).__name__,
        "framework": type(model).__module__.split(".")[0],
        "brainmodelkit_version": __version__,
        "python_version": python_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "save_format": format,
        "target_col": target_col,
        "feature_cols": list(feature_cols) if feature_cols is not None else None,
        "parameters": serializable_parameters(parameters),
    }
    Path(str(local) + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
