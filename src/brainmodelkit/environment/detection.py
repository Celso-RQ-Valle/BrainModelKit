"""Inspect metadata and existing sessions without importing optional packages."""

import platform
import shutil
import sys
from importlib import metadata

PACKAGES = (
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "pyspark",
    "synapseml",
    "mlflow",
    "optuna",
    "Boruta",
    "joblib",
    "cloudpickle",
    "skops",
    "skl2onnx",
    "onnx",
    "onnxruntime",
)


def _probe(callback):
    try:
        return {"value": callback(), "error": None}
    except Exception as exc:
        # Do not expose exception messages, which may contain connection secrets.
        return {"value": None, "error": type(exc).__name__}


def _existing_session():
    # getActiveSession can construct a Python wrapper and set global state.
    # Inspect only already-loaded classes and notebook references instead.
    for name in ("pyspark.sql.session", "pyspark.sql.connect.session"):
        module = sys.modules.get(name)
        cls = vars(module).get("SparkSession") if module else None
        if cls is not None:
            for attribute in ("_activeSession", "_instantiatedSession"):
                session = getattr(cls, attribute, None)
                if session is not None:
                    return session
    main = sys.modules.get("__main__")
    session = vars(main).get("spark") if main else None
    if session is None:
        ipython = sys.modules.get("IPython")
        if ipython is not None:
            shell = ipython.get_ipython()
            if shell is not None:
                session = shell.user_ns.get("spark")
    if session is not None and type(session).__module__.startswith("pyspark."):
        return session
    return None


def detect():
    """Return observations; unavailable or denied checks retain their uncertainty."""
    packages = {name: _probe(lambda n=name: metadata.version(n)) for name in PACKAGES}
    result = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": packages,
        "java_path": _probe(lambda: shutil.which("java")),
        "session": False,
        "runtime": {"value": None, "error": None},
        "java": {"value": None, "error": None},
        "scala": {"value": None, "error": None},
        "synapse_jvm": {"value": None, "error": None},
        "jvm": False,
        "stopped": {"value": None, "error": None},
    }
    found = _probe(_existing_session)
    result["session_error"] = found["error"]
    session = found["value"]
    if session is None:
        return result
    result["session"] = True
    result["runtime"] = _probe(lambda: session.version)

    def stopped():
        context = session._sc
        return context._jsc is None or context._jsc.sc().isStopped()

    result["stopped"] = _probe(stopped)
    jvm = _probe(lambda: session._jvm)
    result["jvm_error"] = jvm["error"]
    if jvm["value"] is None:
        return result
    jvm = jvm["value"]
    result["jvm"] = True
    result["java"] = _probe(lambda: jvm.java.lang.System.getProperty("java.version"))
    result["scala"] = _probe(lambda: jvm.scala.util.Properties.versionNumberString())

    def synapse_class():
        loader = jvm.java.lang.Thread.currentThread().getContextClassLoader()
        # Class.forName with initialize=False does not construct a classifier or
        # trigger its static initializer/native library loading.
        cls = jvm.java.lang.Class.forName(
            "com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassifier", False, loader
        )
        return cls.getName()

    result["synapse_jvm"] = _probe(synapse_class)
    return result
