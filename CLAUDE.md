# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Development Environment Setup
```bash
# Install in editable mode with all dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality and Testing
```bash
# Format code (run before linting)
make format
# OR individually (all tools use 120 character line length):
black --line-length=120 src/icvision tests/
isort --line-length=120 src/icvision tests/

# Run linters
make lint
# OR individually:
flake8 src/icvision tests/
mypy src/icvision/

# Run tests
make test
# OR with coverage:
make coverage
pytest --cov=src/icvision --cov-report=html --cov-report=term tests/

# Run single test
pytest tests/test_core.py::test_specific_function -v

# IMPORTANT: Always run tests before committing
pytest tests/ --tb=short -q
```

### Multi-environment Testing
```bash
# Test across all Python versions
tox

# Run specific environment
tox -e py311
tox -e lint
tox -e type
```

### Build and Package
```bash
# Build package
make build
# OR:
python -m build
```

## Architecture Overview

ICVision is a Python package that automates ICA component classification for EEG data using OpenAI's Vision API. The core workflow follows this pattern:

### Key Components

1. **Core Module (`src/icvision/core.py`)**
   - `label_components()`: Main orchestration function that handles the entire workflow
   - Coordinates data loading, API calls, ICA updates, and artifact rejection
   - Returns cleaned raw data, updated ICA object, and classification results

2. **API Module (`src/icvision/api.py`)**
   - `classify_components_batch()`: Manages parallel OpenAI API requests
   - `classify_component_image_openai()`: Handles individual component classification
   - Uses concurrent.futures for parallel processing with configurable batch sizes

3. **CLI Module (`src/icvision/cli.py`)**
   - Command-line interface with comprehensive argument parsing
   - Entry point: `icvision` command (defined in pyproject.toml)
   - Supports custom prompts, output directories, and all configuration options

4. **Plotting Module (`src/icvision/plotting.py`)**
   - `plot_component_for_classification()`: Creates detailed multi-panel component plots
   - `plot_components_batch()`: Batch processing with enhanced error handling and memory management
   - Supports both individual and batch component visualization with proper cleanup

5. **Configuration (`src/icvision/config.py`)**
   - Contains `DEFAULT_CONFIG` with all default parameters
   - `COMPONENT_LABELS`: Standard classification labels (brain, eye, muscle, heart, line_noise, channel_noise, other_artifact)
   - `OPENAI_ICA_PROMPT`: Default classification prompt for the Vision API

### Data Flow

1. **Input**: Raw EEG data (.set/.fif) + ICA decomposition (.fif)
2. **Visualization**: Generate component plots for OpenAI API
3. **Classification**: Parallel API calls with batching and error handling
4. **Integration**: Update MNE ICA object with labels and exclusions
5. **Output**: Cleaned data, classification results (CSV), summary report (PDF)

### Key Design Patterns

- **Batch Processing**:
  - Uses ThreadPoolExecutor for concurrent API requests
  - Sequential image generation with `plot_components_batch()` and enhanced error handling
  - Proper memory management and cleanup to prevent matplotlib threading issues
- **Error Resilience**: Comprehensive error handling with fallback classifications
- **MNE Integration**: Seamless compatibility with MNE-Python data structures
- **Flexible I/O**: Accepts both file paths and MNE objects as input

## Testing Strategy

Tests are located in `tests/` and use pytest with the following markers:
- `slow`: For tests that take significant time
- `integration`: For end-to-end workflow tests
- `unit`: For isolated unit tests

Mock objects are used extensively for OpenAI API calls to avoid actual API costs during testing.

## Configuration Notes

- Default confidence threshold: 0.8
- Default model: "gpt-4.1"
- Default batch size: 10 components
- Max concurrency: 4 parallel requests
- Auto-exclude all non-brain labels by default
- Line length: 120 characters (Black/flake8)
- Python support: 3.8-3.12

## Git Commit Guidelines

- **NO Claude Attribution**: Never include Claude Code attribution, co-authorship, or generation comments in commit messages
- Keep commit messages professional and focused on the technical changes
- Use conventional commit format when appropriate

## Development Workflow

**CRITICAL**: Before every commit, always run the comprehensive CI test script:

```bash
./test_ci.sh
```

This script runs the exact same checks as tox and GitHub Actions:
1. `pytest tests/ --tb=short -q` - Ensure all tests pass
2. `black --check --diff --line-length=120 src/icvision tests/` - Check formatting
3. `isort --check-only --diff --line-length=120 src/icvision tests/` - Check imports
4. `flake8 src/icvision tests/` - Check linting
5. `mypy --ignore-missing-imports --no-strict-optional --follow-imports=skip src/icvision/` - Check types

If any step fails, the script will exit and show you exactly what needs to be fixed.

**To fix formatting/import issues before committing:**
```bash
black --line-length=120 src/icvision tests/
isort --line-length=120 src/icvision tests/
```

All tools are configured for 120 character line length to avoid conflicts between pre-commit, tox, and manual runs.
