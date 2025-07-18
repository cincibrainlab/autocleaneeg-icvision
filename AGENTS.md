# AGENTS.md

Essential guidelines for coding agents working on ICVision.

## Build/Test Commands
```bash
# Install dev environment
pip install -e ".[dev,test,docs]" && pre-commit install

# Run all CI checks (CRITICAL: run before every commit)
./test_ci.sh

# Run tests
pytest tests/ --tb=short -q                    # Quick test run
pytest tests/test_core.py::test_function -v    # Single test
make test                                       # Full test suite
make coverage                                   # With coverage

# Code quality
make format                                     # Format code (black + isort)
make lint                                       # Run linters (flake8 + mypy)
```

## Code Style (120 char line length)
- **Imports**: Use isort with black profile, group by stdlib/third-party/local
- **Formatting**: Black with 120 char line length, Google docstring convention
- **Types**: Optional typing, use Union/Optional from typing module
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error handling**: Use logging.getLogger(__name__), raise specific exceptions
- **Structure**: Follow src/icvision/ layout, relative imports within package

## Key Patterns
- Use MNE-Python data structures (mne.io.Raw, mne.preprocessing.ICA)
- Batch processing with ThreadPoolExecutor for API calls
- Comprehensive error handling with fallback classifications
- Mock OpenAI API calls in tests to avoid costs