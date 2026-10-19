# Installation, environment checks, and troubleshooting

[Documentation home](README.md) ? [Modeling journeys](../README.md)

- [Requirements](#requirements)
- [Installation](#installation)
- [Environment Doctor](#environment-doctor)
- [Usage](#usage)
- [Spark scalability limits](#spark-scalability-limits)
- [Spark on Windows](#spark-on-windows)

## Requirements

- Python 3.10 or newer

The Python requirement is a packaging minimum, not a guarantee for every Spark
or optional-integration combination. See the [compatibility and release checks](../docs/release-readiness.md)
for the explicit CI matrix, tested scope, and unverified environments.

## Installation

Install the execution stack you need from PyPI:

```bash
# Pandas
python -m pip install "brainmodelkit[pandas]"

# PySpark
python -m pip install "brainmodelkit[pyspark]"

# Pandas + MLflow
python -m pip install "brainmodelkit[pandas,mlflow]"

# PySpark + MLflow
python -m pip install "brainmodelkit[pyspark,mlflow]"
```

`brainmodelkit` by itself installs no execution backend. MLflow is independent
of both backends, so add `[mlflow]` only when model logging or MLflow model
persistence is needed. Persistence formats, ONNX conversion, Boruta, and
Optuna are also separate optional features; see the relevant sections below.

The `[pandas]` extra includes native LightGBM. `[pyspark]` includes SparkLightGBM
(`>=0.9.2,<0.10`); SynapseML is an explicit alternative via `[synapseml]`.
`[full]` is a convenience bundle, not every optional integration: Boruta, skops,
and ONNX still require `[boruta]`, `[secure]`, and `[onnx]`, respectively.

Spark requires a Java installation and a configured Spark environment. The
`[pyspark]` extra requires PySpark 3.5 or newer and includes SparkLightGBM.
For managed Spark, install `[sparklightgbm]` without replacing platform PySpark.
See [Choosing the LightGBM backend](../docs/training.md#choosing-the-lightgbm-backend) for usage.

For a managed environment such as Databricks, use the platform-provided Spark
and MLflow versions when they meet those requirements:

```bash
python -m pip install --no-deps brainmodelkit
```

When installing directly from the repository, use the equivalent command:

```bash
python -m pip install --no-deps "git+https://github.com/Celso-RQ-Valle/BrainModelKit.git"
```

`--no-deps` prevents pip from installing, upgrading, or downgrading runtime
dependencies. Use it only after confirming that the platform already provides
the Pandas, PySpark, selected LightGBM backend, and other optional dependencies required by
the features you will call. In particular, do not use the PySpark extra merely
to replace a managed Spark runtime.

Development tools are available through `BrainModelKit[dev]`. Notebook and
plotting tools such as Jupyter and matplotlib are installed separately.

To use the unreleased source from a local clone, replace the package name with
`.` in the commands above, for example:

```bash
python -m pip install -e ".[pandas]"
```

### SynapseML setup

For SynapseML, install the Python wrapper and load matching JVM artifacts
before the Spark session starts:

```bash
python -m pip install "brainmodelkit[synapseml]"
```

For Spark 4.0.x:

```python
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("BrainModelLightGBM")
    .config(
        "spark.jars.repositories",
        "https://mmlspark.blob.core.windows.net/maven",
    )
    .config(
        "spark.jars.packages",
        "com.microsoft.azure:synapseml_2.13:1.1.3-spark4.0",
    )
    .getOrCreate()
)
```

For Spark 3.5.x, use the compatible SynapseML `_2.12` coordinate documented by
[SynapseML](https://microsoft.github.io/SynapseML/docs/Get%20Started/InstallSynapseML/).
If the Python wrapper imports but the JVM package is missing, training raises
an actionable error instead of the opaque `JavaPackage object is not callable`.

## Environment Doctor

Inspect existing capabilities without changing the environment:

```python
import brainmodelkit as bmk

bmk.doctor()
bmk.doctor(backend="pandas")
bmk.doctor(backend="pyspark")
bmk.doctor(
    backend="pyspark",
    model="lightgbm",
    lgbm_backend="synapseml",
    save_model_to="mlflow",
)
bmk.doctor(verbose=True)
```

Doctor prints a report and returns a dictionary with capabilities, the result,
blocking capabilities, and recommendations. It recommends an available path
based on detected prerequisites, without assuming your workload size. Specify
`backend` when checking a model or persistence destination. Omitted persistence
checks no destination; training itself still defaults to folder persistence.

- `READY`: inspected prerequisites are present and meet package requirements.
- `PARTIAL`: some prerequisites exist, but a runtime check is incomplete or denied.
- `NOT AVAILABLE`: a required dependency is missing.
- `INCOMPATIBLE`: a detected version is below the package's minimum requirement.

Missing optional features affect only workflows that request them. Doctor does
not install anything, change package versions or environment variables, write
test files, change Spark configuration, download JARs, or create/restart Spark.
Package versions come from metadata without importing optional libraries.

In a Databricks notebook with an existing Spark session:

```python
bmk.doctor(verbose=True)
bmk.doctor(backend="pyspark", model="lightgbm", save_model_to="folder", verbose=True)
```

Compatible platform-provided Spark and MLflow installations are preserved.
Doctor inspects existing session references and distinguishes the PySpark Python
package, actual Spark runtime, and SynapseML Python wrapper from the JVM class.
For SynapseML, an installed wrapper alone does not establish JVM readiness.
For SparkLightGBM, both the bridge and native LightGBM must be installed. Restricted JVM
access or Spark Connect produces a partial check, not a recommendation to
replace platform packages. SynapseML artifacts must match the existing
Spark/Scala runtime; meeting `pyspark>=3.5` alone does not certify every pairing.

`READY` is a prerequisite check, not a training or persistence smoke test.
Doctor cannot safely verify native LightGBM execution, credentials, MLflow
tracking/artifact access, or filesystem write permissions. Spark folder writes
remain `PARTIAL` because filesystem/Hadoop support is not exercised. Without
an accessible existing JVM, Java version and SynapseML JVM checks are incomplete;
Doctor only looks for a Java executable and never launches it.

## Usage

```python
import brainmodelkit

print(brainmodelkit.__version__)
```

Integration-specific functionality lives under explicit namespaces:

```python
from brainmodelkit import pandas, pyspark
```

## Spark scalability limits

Spark execution does not establish universal big-data readiness. Current CI
uses small local datasets, not distributed load benchmarks. Standalone overall
metrics use unpartitioned ranking windows; grouped AUC collects distinct group
keys and runs calculations per group. Cube analysis avoids that group-key
collection, but computes `2 ** len(group_columns)` grouping subsets per score.

Feature selection can run many Spark actions and repeated model fits. Correlation
materializes a feature-by-feature matrix on the driver (default limit: 256
features); other selectors collect aggregate summaries. Temporal feature joins
can expand substantially with long histories and many reference dates; exact
median needs group-size memory. Bound feature counts, grouping dimensions, and
history batches, and measure your workload before scaling. Spark Connect and
managed clusters are not covered by release CI; SparkLightGBM requires classic
Spark. Its single-worker mode (including automatic local mode) materializes
training rows in driver NumPy memory; distributed native training requires
`num_workers>1`. See [validation scope and limitations](../docs/release-readiness.md).

## Spark on Windows

Spark can fit successfully and fail when saving. Folder and MLflow destinations
both use Spark's native writer. MLflow may write directly to a remote artifact
filesystem or stage the Spark model through a Hadoop/local filesystem before
uploading it; changing `output_dir`, disabling
`signature`, or dropping the scoring label does not remove this dependency.
See [MLflow Spark persistence](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.spark.html).

An exception mentioning `Shell.checkHadoopHome`, `HADOOP_HOME` or `winutils.exe`
points to Windows Hadoop configuration. For folder persistence on a local
filesystem, BrainModelKit checks the Windows driver helper before fitting when
Hadoop NativeIO is unavailable. Remote filesystems, Linux drivers, and sessions
without a driver JVM do not run this local check. MLflow selects its own save
path without a winutils preflight; actual Hadoop save failures include recovery
instructions. MLflow local staging can still require Windows Hadoop support.
`save_model_to="none"` performs no persistence validation. These checks do not
guarantee filesystem permissions, native-library compatibility or artifact access.

1. Use `save_model_to="none"` to train/evaluate without model persistence while
   configuring the environment. The model remains in `result.model`.
2. For saving, configure Windows Hadoop binaries/native libraries compatible
   with Spark's bundled Hadoop. `HADOOP_HOME` must point to their root containing
   `bin/winutils.exe`; native libraries must be on the process path. Setting a
   variable alone does not install these files.
3. Set up the environment before starting Python/Spark and restart the notebook
   kernel/process. Hadoop caches initialization in the JVM.
4. Alternatively use a configured Linux/WSL Spark environment. A remote MLflow
   server alone does not change the Spark driver's operating system.

Inspect the active classic Spark JVM:

```python
print(spark.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion())
print(spark.sparkContext._jvm.java.lang.System.getProperty("os.name"))
```

See [Hadoop Shell](https://hadoop.apache.org/docs/current/api/org/apache/hadoop/util/Shell.html)
for helper resolution. MLflow database migration and experiment-creation INFO
messages are normal first-use output; inspect the final exception and its cause.
