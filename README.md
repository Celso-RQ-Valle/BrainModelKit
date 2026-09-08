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

## Simulate data

### PySpark credit data

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

## Metrics

### Spark

Install `BrainModelKit[pyspark]`. The examples below use a Spark DataFrame
named `spark_df` with `score`, `target`, and `segment` columns.

#### Kolmogorov-Smirnov (KS)

Calculate approximate KS from score tiles using native Spark operations.
The default is 10 tiles; no Pandas conversion or Python UDF is used:

```python
from brainmodelkit.metrics.pyspark import ks_ntile

overall_ks = ks_ntile(dataframe=spark_df)
segment_ks = ks_ntile(
    dataframe=spark_df,
    group_columns="segment",
    n_tiles=20,
)

overall_ks.show()
segment_ks.show()
```

Defaults are `score_column="score"`, `target="target"`, and
`group_columns=None`. Results contain an uppercase `KS` column and any group
columns. KS is NaN when a dataset or group lacks either target class (0 or 1).
For custom columns, use
`ks_ntile(dataframe=spark_df, score_column="prediction", target="default_flag")`.
Existing positional calls continue to work.

#### ROC AUC and Gini

```python
from brainmodelkit.metrics.pyspark import auc_gini

overall = auc_gini(df=spark_df)
by_segment = auc_gini(df=spark_df, group_by="segment", n_tiles=100)
overall.show()
```

Defaults are `score_column="score"`, `target_column="target"`, and `n_tiles=10`.
Use `score_column` and `target_column` for custom column names, and a list in
`group_by` for multiple grouping columns. Higher scores must indicate class 1.
Targets must be 0 or 1. Null/NaN pairs are removed, and empty or single-class
datasets/groups return NaN. Output columns are `auc` and `gini` plus group keys.
Gini is `2 * auc - 1`.

Spark AUC uses a score-tile approximation; it can differ from exact Pandas AUC.
Equal scores can span tiles, making results sensitive to their ordering.
Overall ranking uses an unpartitioned window. Grouped calculations collect
distinct group keys and compute each group separately, so use a modest number
of groups. The function runs Spark jobs immediately without collecting input rows.

### Pandas

Install `BrainModelKit[pandas]` to include Pandas and scikit-learn.
Create a sample DataFrame for the examples below:

```python
import pandas as pd

df = pd.DataFrame(
    {
        "score": [0.9, 0.8, 0.2, 0.1],
        "target": [1, 1, 0, 0],
        "segment": ["A", "B", "A", "B"],
    }
)
```

#### Kolmogorov-Smirnov (KS)

Calculate exact point-by-point KS overall or by group:

```python
from brainmodelkit.metrics.pandas import ks

overall_ks = ks(dataframe=df)
segment_ks = ks(dataframe=df, group_columns="segment")
print(overall_ks)  # KS = 1.0
```

Defaults are `score_column="score"`, `target="target"`, and
`group_columns=None`. Results contain an uppercase `KS` column and any group
columns. KS is NaN when a dataset or group lacks either target class (0 or 1).
For custom columns, use
`ks(dataframe=df, score_column="prediction", target="default_flag")`.
Existing positional calls continue to work. The lower-level
`calculate_ks(y_true, y_pred)` function accepts two actual data series.

#### ROC AUC and Gini

```python
from brainmodelkit.metrics.pandas import auc_gini

overall = auc_gini(df=df)
by_segment = auc_gini(df=df, group_by="segment")
print(overall)  # auc = 1.0, gini = 1.0
```

Defaults are `score_column="score"`, `target_column="target"`, and `group_by=None`.
Results contain lowercase `auc` and `gini` columns, plus any grouping columns.
Use `score_column` and `target_column` for custom column names, and a list for
multiple grouping columns. Higher scores must indicate the positive class.
Missing target/score pairs are dropped; empty data or single-class groups
produce NaN. Gini is `2 * auc - 1`.

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
