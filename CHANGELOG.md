# Changelog

All notable changes to BrainModelKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Cube-only performance improvements: Spark computes grouped AUC and n-tile
  tables with partitioned windows, avoids group-key collection, and validates
  once per call. Pandas reuses grouping across scores and avoids metric merges.

- Cube functions use `group_columns`, `score_columns`, and `target_column`
  keywords in place of `grupos`, `scores`, and `target`; positional order is
  unchanged. Grouping is optional, and README cube examples start with Spark.

- Restored `auc_gini` as the primary name; `roc_auc_gini` remains an alias.
- README Spark examples explicitly show the default `n_tiles=10`.

### Added

- Complete Spark feature-selection namespace: binned mutual information, ANOVA,
  permutation importance, fold-local RFECV, forward/backward sequential selection,
  Poisson-bootstrap stability, and native forest Boruta with corrected hit tests.
- Per-method algorithm/parameter guides, Boruta inference documentation, complete
  README quickstarts, and a standalone selector example runner for both backends.
- Spark extra now includes SciPy for scalar F/binomial probability calculations;
  distributed datasets are not converted to Pandas or passed to sklearn.

- Pandas and Spark `cube_analysis.calculate_ntile` functions for risk tables
  within every grouping subset and score.

- `cube_analysis.pandas` and `cube_analysis.pyspark` with `calculate_metrics` for
  all grouping subsets and multiple scores, five-decimal metrics, and `Geral` rollups.

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
