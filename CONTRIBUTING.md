# Contributing to BrainModelKit

Thank you for helping improve BrainModelKit. The project is in an early stage,
so please open an issue before starting a large change or introducing a new
public abstraction.

## Development setup

Fork and clone the repository, then create a virtual environment:

```bash
python -m venv .venv
```

Activate it and install the project in editable mode:

```bash
python -m pip install -e ".[dev]"
```

To work on an integration, include its extra:

```bash
python -m pip install -e ".[dev,pandas]"
python -m pip install -e ".[dev,pyspark]"
```

## Quality checks

Run these commands before submitting a pull request:

```bash
ruff check .
ruff format --check .
pytest
python -m build
```

Ruff can apply safe lint fixes and format the code:

```bash
ruff check --fix .
ruff format .
```

## Code guidelines

- Add type hints to public functions and methods.
- Write public names, docstrings, comments, and documentation in English.
- Keep the core package independent of Pandas and PySpark.
- Put framework-specific behavior in its corresponding integration namespace.
- Import optional dependencies only inside the integration that needs them.
- Add tests for new behavior and bug fixes.
- Keep the public API small and document compatibility-impacting changes.

## Pull requests

Use a focused title and describe the motivation, implementation, and test
coverage. Keep unrelated changes in separate pull requests. By contributing,
you agree that your contribution will be licensed under the project's MIT
License.
