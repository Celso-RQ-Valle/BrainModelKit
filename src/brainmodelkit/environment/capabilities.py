"""Centralized conservative rules, aligned with pyproject.toml requirements."""

import re

READY = "READY"
PARTIAL = "PARTIAL"
MISSING = "NOT AVAILABLE"
INCOMPATIBLE = "INCOMPATIBLE"
# Packaging lower bounds, not a certification of future runtime releases.
MINIMUMS = {
    "python": "3.10",
    "pandas": "2.0",
    "numpy": "1.23",
    "scipy": "1.9",
    "scikit-learn": "1.2",
    "lightgbm": "4.0",
    "pyspark": "3.5",
    "mlflow": "2.12",
    "optuna": "3.0",
    "Boruta": "0.4.3",
    "joblib": "1.2",
    "cloudpickle": "2.2",
    "skops": "0.10",
    "skl2onnx": "1.16",
    "onnx": "1.16",
    "onnxruntime": "1.17",
}
SYNAPSE_VERSION = "1.1.3"


def version_status(value, minimum):
    """Compare final releases; unfamiliar and prerelease forms are unknown."""
    if value is None:
        return MISSING
    match = re.fullmatch(r"(\d+(?:\.\d+)*)(?:\+[-\w.]+)?", str(value))
    if not match:
        return PARTIAL
    actual = tuple(int(part) for part in match[1].split("."))
    required = tuple(int(part) for part in minimum.split("."))
    length = max(len(actual), len(required))
    return (
        READY
        if actual + (0,) * (length - len(actual))
        >= (required + (0,) * (length - len(required)))
        else INCOMPATIBLE
    )


def combine(statuses):
    for status in (INCOMPATIBLE, MISSING, PARTIAL):
        if status in statuses:
            return status
    return READY


def evaluate(env):
    """Evaluate independent capabilities without probing or changing the environment."""
    packages = {}
    for name, observation in env["packages"].items():
        value = observation["value"]
        if value is None:
            packages[name] = (
                MISSING if observation["error"] == "PackageNotFoundError" else PARTIAL
            )
        elif name == "synapseml":
            packages[name] = READY if value == SYNAPSE_VERSION else PARTIAL
        else:
            packages[name] = version_status(value, MINIMUMS[name])
    python = version_status(env["python"], MINIMUMS["python"])

    def stack(*names):
        return combine([python, *(packages[name] for name in names)])

    runtime = version_status(env["runtime"]["value"], MINIMUMS["pyspark"])
    if runtime == MISSING:
        runtime = PARTIAL
    spark = combine([stack("pyspark", "numpy", "scipy"), runtime])
    if spark == READY and (
        not env["jvm"]
        or not env["java"]["value"]
        or env["stopped"]["value"] is not False
    ):
        spark = PARTIAL  # Current Spark ML code requires classic JVM access.
    if spark == READY:
        package_minor = env["packages"]["pyspark"]["value"].split(".")[:2]
        runtime_minor = env["runtime"]["value"].split(".")[:2]
        if package_minor != runtime_minor:
            spark = PARTIAL  # Mismatched client/server interoperability is unverified.
    lightgbm = combine([spark, packages["synapseml"]])
    if lightgbm == READY and not env["synapse_jvm"]["value"]:
        lightgbm = PARTIAL
    caps = {
        "Python": python,
        "Pandas backend": stack("pandas", "numpy", "scipy", "scikit-learn"),
        "PySpark backend": spark,
        "Pandas LightGBM": stack(
            "pandas", "numpy", "scipy", "scikit-learn", "lightgbm"
        ),
        "Spark LightGBM": lightgbm,
        "Folder persistence": READY,
        "Spark folder persistence": PARTIAL,
        "MLflow persistence": stack("mlflow"),
        "ONNX persistence": stack("skl2onnx", "onnx", "onnxruntime"),
        "Optuna": stack("optuna"),
        "Boruta": stack("Boruta"),
        "joblib": stack("joblib"),
        "cloudpickle": stack("cloudpickle"),
        "skops": stack("skops"),
    }
    return packages, caps
