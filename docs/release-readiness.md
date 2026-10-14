# Compatibility and release validation

BrainModelKit is experimental, pre-1.0 software with unstable public APIs.
Dependency minimums and Python classifiers express install metadata; they do
not certify every combination accepted by pip.

## CI compatibility matrix

The shared CI workflow requires the full test suite and clean-wheel lifecycle
checks for these explicit pairs on Ubuntu with Temurin Java 17:

| Python | PySpark | Validation scope |
| --- | --- | --- |
| 3.10 | 3.5.7 | Full suite, required integrations, Pandas and Spark wheel checks |
| 3.11 | 3.5.7 | Full suite, required integrations, Pandas and Spark wheel checks |
| 3.12 | 4.0.2 | Full suite, required integrations, Pandas and Spark wheel checks |
| 3.13 | 4.0.2 | Full suite, required integrations, Pandas and Spark wheel checks |

These are configured CI targets. A pair is **tested for a commit only when its
CI job passes**; this document does not claim that newly configured jobs have
already passed. Inspect the commit's CI results and uploaded JUnit reports.
Other Python/Spark pairs, operating systems, Java versions, Spark Connect,
remote storage, and managed/multi-node clusters are unverified by this matrix.
The choice of Java 17 is consistent with the upstream
[Spark 3.5 installation requirements](https://spark.apache.org/docs/3.5.7/api/python/getting_started/install.html)
and [Spark 4.0 requirements](https://spark.apache.org/docs/4.0.0/index.html).

CI explicitly installs `dev,pandas,optimization,mlflow,boruta,persistence` and
pins PySpark separately. Optuna optimization, Pandas Boruta, MLflow sklearn
logging/reloading, and SparkLightGBM native/MLflow persistence tests must run.
`--fail-on-skip` turns both collection-time and runtime skips into failures.
Ordinary local pytest runs retain optional-dependency skips. `pip check` catches
resolver conflicts, and `pip freeze` records the resolved integration versions
in each job log. Other dependencies track the declared ranges, not a lockfile.

SynapseML backend selection and missing-JVM diagnostics have contract tests;
a real SynapseML training run with matching Maven artifacts is **not** covered.
SparkLightGBM runs in classic local Spark. ONNX and skops have mocked contract
coverage, not installed end-to-end coverage. These distinctions also apply to
release validation; installing a wrapper is not proof of JVM/backend readiness.

## Built-wheel checks

`python -m build` builds an sdist and a wheel from that sdist; `twine check`
validates distribution metadata. `scripts/check_wheel.py` creates a separate
virtual environment for each execution backend with no system site-packages.
It first installs only the wheel and verifies that importing the base package
needs none of Pandas, Spark, MLflow, Optuna, or Boruta. It then installs that
wheel's backend extra, checks dependency consistency, and runs outside the
checkout with Python isolated mode (`-I`) and no `PYTHONPATH`.

The smoke script verifies that imports resolve inside the fresh environment.
Each backend trains logistic regression, evaluates a distinct OOT dataset,
writes a real folder model and metrics, reloads through `load_model`, and checks
predictions. Spark reloads the raw-feature PipelineModel and compares both
probabilities and class predictions. There are no mocks or skip conditions.
These checks cover default folder persistence; MLflow and SparkLightGBM are
covered separately in the full integration suite.

To reproduce after building (requires dependency-download access):

```bash
python -m build
python -m twine check dist/*
python scripts/check_wheel.py --spark-version 4.0.2
```

Use `--backend pandas` or `--backend spark` to check one backend. Keep exactly
one wheel in `dist/` (or select another directory with `--dist-dir`). Spark
requires Java and working native filesystem support;
Windows without Hadoop support cannot complete its disk-persistence check.

## Release gate

On a published-release event, the publishing workflow checks out the event's
commit SHA (not a mutable tag) and passes that SHA to the shared CI workflow. Every checkout,
lint check, source test, build, and wheel test uses that commit. Publishing
requires the entire validation workflow to succeed and downloads its build
artifact from the same workflow run; it does not rebuild or rely on an older
main-branch status. Failed, cancelled, or skipped validation prevents publishing.
The PyPI environment retains its configured approval/protection rules.

Changing these workflows does not publish a package or create a release/tag.

## Scale and operational limits

The tests use small datasets on local Spark. They do not measure throughput,
executor failures, storage credentials, or production cluster capacity.
Standalone overall metrics have unpartitioned ranking windows; grouped AUC
collects keys and evaluates groups separately. Cube analysis uses partitioned
windows for groups but enumerates all grouping subsets for every score.
Correlation collects a quadratic feature matrix on the driver, and iterative
selectors/CV/optimization can trigger many fits, scans, and shuffles. Temporal
joins expand with history length and reference-date count; exact median has
per-group memory costs. See the README and temporal guide for practical bounds.
SparkLightGBM's single-worker mode materializes training rows in driver NumPy
memory; `num_workers>1` is required for distributed native training.
No universal big-data readiness or managed-platform certification is implied.
