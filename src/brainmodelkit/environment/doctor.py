"""Human-readable, operation-aware environment diagnostics."""

from .capabilities import MINIMUMS, PARTIAL, READY, SYNAPSE_VERSION, combine, evaluate
from .detection import detect


def doctor(*, backend=None, model=None, save_model_to=None, verbose=False):
    """Print and return a diagnostic dictionary without changing the environment.

    READY means the inspected prerequisites are present, not that a training
    job, native library, artifact store, or filesystem write has been tested.
    Omitted persistence checks no destination; use folder/mlflow/none explicitly.
    A model or persistence request requires an explicit backend.
    """
    if backend not in (None, "pandas", "pyspark"):
        raise ValueError("backend must be pandas, pyspark, or None")
    if model not in (
        None,
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "lightgbm",
    ):
        raise ValueError("model must be a supported training model name or None")
    if save_model_to not in (None, "folder", "mlflow", "none"):
        raise ValueError("save_model_to must be folder, mlflow, none, or None")
    if backend is None and (model is not None or save_model_to is not None):
        raise ValueError("Specify backend when requesting a model or persistence")

    from brainmodelkit import __version__

    env = detect()
    packages, caps = evaluate(env)
    recommendations = []
    if packages["pyspark"] == READY:
        recommendations.append(
            f"Keep the existing PySpark installation: it satisfies >="
            f"{MINIMUMS['pyspark']}. A lower bound is not certification of "
            "every Spark/SynapseML runtime combination."
        )
    if packages["mlflow"] == READY:
        recommendations.append(
            "Preserve the existing MLflow installation; its version meets "
            "BrainModelKit requirements."
        )
    else:
        recommendations.append(
            "MLflow is independent: its absence does not affect normal training. "
            "Only MLflow persistence/tracking requires it."
        )
    selected = []
    path = None
    if backend:
        selected = [
            "Python",
            "Pandas backend" if backend == "pandas" else "PySpark backend",
        ]
        if model == "lightgbm":
            selected.append(
                "Pandas LightGBM" if backend == "pandas" else "Spark LightGBM"
            )
        if save_model_to == "mlflow":
            selected.append("MLflow persistence")
        elif save_model_to == "folder":
            selected.append(
                "Folder persistence"
                if backend == "pandas"
                else "Spark folder persistence"
            )
    else:
        if caps["PySpark backend"] == READY:
            path = "Existing Spark environment"
            selected = ["PySpark backend"]
            recommendations.append(
                "An existing runtime satisfies Spark prerequisites; preserve it."
            )
        elif caps["Pandas backend"] == READY:
            path = "Pandas"
            selected = ["Pandas backend"]
            recommendations.append(
                "Pandas prerequisites are already available. Installing Spark is "
                "unnecessary unless distributed processing is required."
            )
        else:
            path = "No verified backend"
            selected = ["Pandas backend", "PySpark backend"]
            recommendations.append(
                "Choose the backend your workflow needs. If dependencies are missing, "
                "use pip install 'brainmodelkit[pandas]' or 'brainmodelkit[pyspark]' "
                "in a suitable environment; preserve managed runtime packages."
            )
    blockers = {name: caps[name] for name in selected if caps[name] != READY}
    if "PySpark backend" in blockers or "Spark LightGBM" in blockers:
        recommendations.append(
            "Inspect the existing Spark runtime and JVM access: package metadata "
            "alone cannot validate Spark execution. No session was created. "
            "Connect/restricted JVM sessions cannot fully validate current Spark ML "
            "code. Stopped sessions or differing client/runtime major-minor versions "
            "also require review."
        )
    if "Spark LightGBM" in blockers:
        recommendations.append(
            "Spark LightGBM requires both the SynapseML Python wrapper and JVM "
            "integration. Ask the runtime administrator to verify the SynapseML "
            "artifact for the existing Spark/Scala runtime; do not replace compatible "
            "PySpark to satisfy SynapseML. Class availability does not test native "
            "LightGBM."
        )
    if "MLflow persistence" in blockers:
        recommendations.append(
            "MLflow blocks the requested persistence. If absent, install optional "
            "support with pip install 'brainmodelkit[mlflow]'; otherwise review "
            "the detected version with the environment administrator."
        )
    if "Pandas backend" in blockers or "Pandas LightGBM" in blockers:
        recommendations.append(
            "Review the missing/unsupported Pandas prerequisites below; optional "
            "support is available via pip install 'brainmodelkit[pandas]'."
        )
    if "Spark folder persistence" in blockers:
        recommendations.append(
            "Spark folder persistence needs a configured Hadoop filesystem. "
            "Write permissions/native filesystem support cannot be verified read-only."
        )
    result = combine(list(blockers.values())) if selected else PARTIAL
    report = {
        "version": __version__,
        "environment": env,
        "packages": packages,
        "capabilities": caps,
        "result": result,
        "blocking_capabilities": blockers,
        "recommended_path": path,
        "recommendations": recommendations,
    }
    print("BrainModelKit Doctor\n====================")
    print(f"BrainModelKit {__version__} | Python {env['python']} | {env['platform']}")
    if backend:
        print(
            f"\nRequested operation: backend={backend}, model={model or 'default'}, "
            f"persistence={save_model_to or 'not requested'}"
        )
    else:
        print(f"\nRecommended path: {path}")
    print("\nEnvironment (package metadata; optional packages are not imported)")
    for name, observation in env["packages"].items():
        print(f"{name:24} {observation['value'] or 'not detected':18} {packages[name]}")
    print(
        f"Existing Spark session   {'detected' if env['session'] else 'not detected'}"
    )
    print(f"Spark runtime            {env['runtime']['value'] or 'not validated'}")
    print(f"Java                     {env['java']['value'] or 'version not validated'}")
    java_executable = "found" if env["java_path"]["value"] else "not detected"
    print(f"Java executable          {java_executable}")
    print(f"Scala                    {env['scala']['value'] or 'not validated'}")
    print(
        f"SynapseML JVM            {READY if env['synapse_jvm']['value'] else PARTIAL}"
    )
    print("\nCapabilities")
    for name, status in caps.items():
        print(f"{name:26} {status}")
    print(f"\nResult: {result}")
    for name, status in blockers.items():
        print(f"Blocking capability: {name} ({status})")
    if result == READY:
        print("No environment changes are required for the inspected prerequisites.")
    print("\nRecommendations")
    for recommendation in recommendations:
        print(f"- {recommendation}")
    print(
        "\nRead-only: no installs, version changes, Spark creation, or test writes. "
        "READY describes prerequisites, not an executed workflow. "
        "Filesystem permissions, MLflow connectivity and native libraries are untested."
    )
    if verbose:
        print("\nTechnical observations (errors are redacted to exception types)")
        print(
            "Package lower bounds: "
            + ", ".join(f"{name}>={version}" for name, version in MINIMUMS.items())
        )
        print(
            f"SynapseML wrapper expected by packaging: {SYNAPSE_VERSION}; "
            "other versions remain unverified, not a recommendation to downgrade."
        )
        print(f"Java executable: {env['java_path']['value'] or 'unknown'}")
        for name, observation in {
            **env["packages"],
            **{
                key: env[key]
                for key in ("runtime", "java", "scala", "synapse_jvm", "stopped")
            },
        }.items():
            if observation["error"]:
                print(f"{name}: {observation['error']}")
        print(f"Session lookup: {env['session_error'] or 'completed'}")
        print(f"JVM lookup: {env.get('jvm_error') or 'no error reported'}")
    return report
