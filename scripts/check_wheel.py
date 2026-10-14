"""Install the built wheel in isolated environments and run lifecycle checks."""

import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spark-version", default="4.0.2")
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--backend", choices=["pandas", "spark", "all"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wheels = list(args.dist_dir.resolve().glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit("Build exactly one wheel in dist/ before running this check")
    backends = ["pandas", "spark"] if args.backend == "all" else [args.backend]
    for backend in backends:
        with tempfile.TemporaryDirectory(prefix="bmk-wheel-") as directory:
            work = Path(directory)
            environment = work / "venv"
            venv.EnvBuilder(with_pip=True).create(environment)
            python = environment / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            env["PYSPARK_PYTHON"] = str(python)
            env["PYSPARK_DRIVER_PYTHON"] = str(python)
            env.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

            def run(*arguments, python=python, work=work, env=env):
                subprocess.run(
                    [str(python), "-I", *arguments], cwd=work, env=env, check=True
                )

            run("-m", "pip", "install", "--no-deps", str(wheels[0]))
            run(
                "-c",
                "import brainmodelkit, importlib.util; "
                "assert all(importlib.util.find_spec(m) is None "
                "for m in ('pandas', 'pyspark', 'mlflow', 'optuna', 'boruta'))",
            )
            extra = "pandas" if backend == "pandas" else "pyspark"
            dependencies = (
                [] if backend == "pandas" else [f"pyspark=={args.spark_version}"]
            )
            run("-m", "pip", "install", f"{wheels[0]}[{extra}]", *dependencies)
            run("-m", "pip", "check")
            run("-m", "pip", "freeze")
            run(str(root / "scripts" / "wheel_smoke.py"), backend)


if __name__ == "__main__":
    main()
