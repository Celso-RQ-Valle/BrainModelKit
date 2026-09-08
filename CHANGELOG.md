# Changelog

All notable changes to BrainModelKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Restored `auc_gini` as the primary name; `roc_auc_gini` remains an alias.
- README Spark examples explicitly show the default `n_tiles=10`.

### Added

- Pandas and Spark `risk_table` with score ranges, volume, event counts, and rates.

- Pandas and Spark `curve_roc` DataFrames with README plotting examples.

- Spark `roc_auc_gini` metrics with configurable score tiles and optional grouping.
- Pandas `roc_auc_gini` metrics with default column names and optional grouping.
- Scikit-learn dependency in the Pandas, dataframes, and development extras.

## [0.2.0] - 2026-09-07

### Changed

- KS functions default to `score` and `target` columns with no grouping.
- Existing positional calls remain supported; missing inputs raise clear errors.
- Documented minimal calls and a complete Pandas example.

### Added

- Synthetic PySpark credit data generation with configurable model features.
- Exact point-by-point KS metrics for Pandas.
- Native PySpark KS metrics based on configurable score tiles.
- Automated packaging, test, and publishing workflows.

## [0.1.0] - 2026-09-05

### Added

- Initial project structure and optional Pandas and PySpark integrations.

[Unreleased]: https://github.com/Celso-RQ-Valle/BrainModelKit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Celso-RQ-Valle/BrainModelKit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Celso-RQ-Valle/BrainModelKit/releases/tag/v0.1.0
