# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and core functionality for ICVision.
- Support for loading MNE Raw and ICA data (objects or file paths).
- ICA component classification using OpenAI Vision API.
  - Generation of standardized component plots (topography, time series, PSD, ERP image).
  - Batch processing of components with parallel API requests.
- Configuration options for API model, prompt, confidence thresholds, labels to exclude.
- Updating MNE ICA object with classification labels and exclusion flags.
- Applying artifact rejection to Raw data based on classifications.
- Saving results: cleaned Raw data, updated ICA object, CSV of classifications, text summary.
- Generation of PDF reports with classification summaries and detailed component plots.
- Command-line interface (CLI) for main functionality.
- Python API for programmatic use (`label_components` function).
- Utility functions for data loading, validation, output management.
- Plotting functions for component visualization.
- API interaction module for OpenAI communication.
- Comprehensive unit and integration tests (Pytest).
- `pyproject.toml` for package configuration with Hatchling.
- Development tools: Black, MyPy, Flake8, Pytest, pre-commit, tox.
- Initial `README.md`, `LICENSE` (MIT), `CHANGELOG.md`, `CONTRIBUTING.md`.
- Basic `.gitignore`.

### Changed
- Precompute ICA sources in batch plotting to dramatically speed up plot generation.

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - YYYY-MM-DD

- Placeholder for the first official release.
