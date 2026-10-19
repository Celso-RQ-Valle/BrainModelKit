# Development and project reference

[Documentation home](README.md) ? [Modeling journeys](../README.md)

- [Development](#development)
- [Project layout](#project-layout)
- [License](#license)

## Development

Create a virtual environment and install the development dependencies:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Release CI explicitly adds optional integrations and rejects skipped tests:

```bash
python -m pip install -e ".[dev,pandas,optimization,mlflow,boruta,persistence]" "pyspark==4.0.2"
python -m pytest --fail-on-skip
```

Use a Python/Spark pair from the [CI matrix](../docs/release-readiness.md).
Local tests without the extra dependencies can skip optional integrations;
Windows without Hadoop support can skip Spark disk round trips. Neither counts
as release validation. CI also builds the distributions and runs
`python scripts/check_wheel.py --spark-version 4.0.2` in fresh virtual environments.

For local notebook development, install the notebook
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

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the complete contribution guide.

## Project layout

```text
src/brainmodelkit/       Library source code
tests/                   Automated tests
pyproject.toml           Build and tool configuration
```

## License

BrainModelKit is distributed under the MIT License. See [LICENSE](../LICENSE).
