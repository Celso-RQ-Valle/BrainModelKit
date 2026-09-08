# BrainModelKit

BrainModelKit is an early-stage Python library for building, evaluating, and
analyzing models. Its package structure is ready for optional Pandas and
PySpark integrations without requiring either framework for core usage.

> [!WARNING]
> The public API is not stable yet. BrainModelKit is currently under active
> development.

## Requirements

- Python 3.10 or newer

## Installation

Install the core package from PyPI:

```bash
python -m pip install BrainModelKit
```

Install an optional dataframe integration from PyPI:

```bash
python -m pip install "BrainModelKit[pandas]"
python -m pip install "BrainModelKit[pyspark]"
```

Install both integrations:

```bash
python -m pip install "BrainModelKit[dataframes]"
```

To use the unreleased source from a local clone, replace `BrainModelKit` with
`.` in the commands above, for example:

```bash
python -m pip install ".[pyspark]"
```

## Usage

```python
import brainmodelkit

print(brainmodelkit.__version__)
```

Integration-specific functionality lives under explicit namespaces:

```python
from brainmodelkit import pandas, pyspark
```

### Simulate credit data with PySpark

In a Jupyter notebook, install the package into the active kernel and restart
the kernel if prompted:

```python
%pip install "BrainModelKit[pyspark]"
```

Generate 10,000 rows with the default 20 numeric features:

```python
from brainmodelkit.pyspark import simulate_credit_data

credit_df, feature_columns = simulate_credit_data()

credit_df.show(10, truncate=False)
print(feature_columns)
```

You can pass an existing Spark session and change the dataset dimensions. This
example creates 5,000 rows and 30 numeric features:

```python
from pyspark.sql import SparkSession

from brainmodelkit.pyspark import simulate_credit_data

spark = SparkSession.builder.appName("CreditModelNotebook").getOrCreate()
credit_df, feature_columns = simulate_credit_data(
    spark,
    row_count=5_000,
    feature_count=30,
    max_days=30,
    seed=42,
)
```

The returned DataFrame contains `company_id`, `reference_date`, `status`,
`default_flag`, `industry_section`, `company_size`, and the requested
`feature_XX` columns. Company size is one of `Small`, `Medium`, `Large`, or
`Very Large` and contributes to the simulated default risk.
Use `default_flag` as the binary training label. `feature_columns` contains the
numeric input column names for a Spark ML `VectorAssembler`:

```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
model_data = assembler.transform(credit_df).select(
    "features",
    "default_flag",
)
train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)
```

### Kolmogorov-Smirnov metrics

Calculate exact point-by-point KS with Pandas, either overall or by group:

```python
from brainmodelkit.metrics.pandas import ks

overall_ks = ks(dataframe=pandas_df)
segment_ks = ks(dataframe=pandas_df, group_columns="segment")
```

For large Spark datasets, calculate an approximate KS from score tiles using
only native Spark operations. The default is 10 tiles; no Pandas conversion or
Python UDF is used:

```python
from brainmodelkit.metrics.pyspark import ks_ntile

# Overall decile KS.
overall_ks = ks_ntile(dataframe=spark_df)

# KS by segment with a custom number of tiles.
segment_ks = ks_ntile(
    dataframe=spark_df,
    group_columns="segment",
    n_tiles=20,
)

overall_ks.show()
segment_ks.show()
```

Both functions return a DataFrame containing an uppercase `KS` column. Grouped
calculations also include the requested group columns. KS is `NaN` when a
dataset or group does not contain both target classes (`0` and `1`).

The defaults are `score_column="score"`, `target="target"`, and
`group_columns=None`. Supply your DataFrame using `dataframe=...`; data cannot
be inferred. For other column names, use
`ks(dataframe=pandas_df, score_column="prediction", target="default_flag")`.
Existing positional calls continue to work. `calculate_ks(y_true, y_pred)`
requires the two actual data series.

A complete Pandas example:

```python
import pandas as pd
from brainmodelkit.metrics.pandas import ks

df = pd.DataFrame({"score": [0.9, 0.8, 0.2, 0.1], "target": [1, 1, 0, 0]})
print(ks(dataframe=df))  # KS = 1.0
```

## Development

Create a virtual environment and install the development dependencies:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

For local notebook development, install the small, reproducible notebook
dependency set:

```bash
python -m pip install -r requirements-notebooks.txt
```

Run the quality checks:

```bash
pytest
ruff check .
ruff format --check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete contribution guide.

## Project layout

```text
src/brainmodelkit/       Library source code
tests/                   Automated tests
pyproject.toml           Build and tool configuration
```

## License

BrainModelKit is distributed under the MIT License. See [LICENSE](LICENSE).
