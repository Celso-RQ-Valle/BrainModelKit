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

Install the core package from the repository:

```bash
python -m pip install .
```

Install an optional dataframe integration:

```bash
python -m pip install ".[pandas]"
python -m pip install ".[pyspark]"
```

Install both integrations:

```bash
python -m pip install ".[dataframes]"
```

## Usage

```python
import brainmodelkit

print(brainmodelkit.__version__)
```

Integration-specific functionality will live under explicit namespaces:

```python
from brainmodelkit import pandas, pyspark
```

## Development

Create a virtual environment and install the development dependencies:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
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
