.PHONY: help install install-dev lint format test test-openai-gate0 test-openai-gate1 test-openai-contracts test-openai-compat test-openai-modernization coverage clean build publish docs docs-clean docs-live

# Variables
PYTHON = python3
OPENAI_PYTHON = python
PIP = $(PYTHON) -m pip
VENV_DIR = .venv
SRC_DIR = src/icvision
TEST_DIR = tests

# Default target executed when no arguments are given to make.
help:
	@echo "ICVision Makefile"
	@echo "-----------------"
	@echo "Available targets:"
	@echo "  install         Install the package in the current environment."
	@echo "  install-dev     Install the package in editable mode with development dependencies."
	@echo "  lint            Run linters (flake8, mypy)."
	@echo "  format          Format code (black, isort)."
	@echo "  test            Run tests (pytest)."
	@echo "  test-openai-gate0 Run exact offline Gate 0 policy tests."
	@echo "  test-openai-gate1 Run exact offline Gate 1 contract tests."
	@echo "  test-openai-contracts Run cumulative offline Gate 0 and Gate 1 tests."
	@echo "  test-openai-compat Run affected legacy SDK/API compatibility tests."
	@echo "  test-openai-modernization Run Gate 0, Gate 1, and compatibility tests."
	@echo "  coverage        Run tests and generate a coverage report."
	@echo "  clean           Remove build artifacts, bytecode, and cache files."
	@echo "  build           Build the package (sdist and wheel)."
	@echo "  publish         Publish the package to PyPI (requires twine and credentials)."
	@echo "  docs            Build documentation with Sphinx."
	@echo "  docs-clean      Clean documentation build files."
	@echo "  docs-live       Start live documentation server with auto-reload."

# Installation
install:
	@echo "Installing ICVision..."
	$(PIP) install .

install-dev:
	@echo "Installing ICVision in editable mode with development dependencies..."
	$(PIP) install -e ".[dev,test,docs]"
	@echo "Installing pre-commit hooks..."
	pre-commit install

# Code Quality
lint: format # Ensure code is formatted before linting for fewer errors
	@echo "Running linters..."
	flake8 $(SRC_DIR) $(TEST_DIR)
	mypy $(SRC_DIR)

format:
	@echo "Formatting code..."
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)

# Testing
test:
	@echo "Running tests..."
	pytest $(TEST_DIR)


test-openai-gate0:
	$(OPENAI_PYTHON) -B -m pytest tests/test_transport_policy.py -q --maxfail=1 --no-cov -p no:cacheprovider

test-openai-gate1:
	$(OPENAI_PYTHON) -B -m pytest tests/test_request_contracts.py -q --maxfail=1 --no-cov -p no:cacheprovider

test-openai-contracts:
	$(OPENAI_PYTHON) -B -m pytest tests/test_transport_policy.py tests/test_request_contracts.py -q --maxfail=1 --no-cov -p no:cacheprovider

test-openai-compat:
	$(OPENAI_PYTHON) -B -m pytest tests/test_core.py tests/test_phase3_cli_api.py tests/test_phase4_retry.py tests/test_strip_compatibility.py --deselect=tests/test_core.py::test_label_components_custom_params -q --maxfail=1 --no-cov -p no:cacheprovider

test-openai-modernization:
	$(OPENAI_PYTHON) -B -m pytest tests/test_transport_policy.py tests/test_request_contracts.py tests/test_core.py tests/test_phase3_cli_api.py tests/test_phase4_retry.py tests/test_strip_compatibility.py --deselect=tests/test_core.py::test_label_components_custom_params -q --maxfail=1 --no-cov -p no:cacheprovider

coverage:
	@echo "Running tests and generating coverage report..."
	pytest --cov=$(SRC_DIR) --cov-report=html --cov-report=term $(TEST_DIR)
	@echo "Coverage report generated in htmlcov/"

# Cleaning
clean:
	@echo "Cleaning up build artifacts, bytecode, and cache files..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage .cache .tox $(VENV_DIR)
	@echo "Clean complete. Virtual environment $(VENV_DIR) has been removed if it existed."

# Building and Publishing (Example - adjust as needed)
build:
	@echo "Building package (sdist and wheel)..."
	$(PYTHON) -m build

publish: build
	@echo "Publishing package to PyPI... (Ensure you have Twine and credentials)"
	$(PYTHON) -m twine upload dist/*

# Documentation
docs:
	@echo "Building documentation with Sphinx..."
	@if [ -d "docs" ] && [ -f "docs/Makefile" ]; then \
		$(MAKE) -C docs html; \
		@echo "Documentation built in docs/_build/html/"; \
	else \
		@echo "Error: docs directory or docs/Makefile not found."; \
		exit 1; \
	fi

docs-clean:
	@echo "Cleaning documentation build files..."
	@if [ -d "docs" ] && [ -f "docs/Makefile" ]; then \
		$(MAKE) -C docs clean; \
	fi

docs-live:
	@echo "Starting live documentation server..."
	@if [ -d "docs" ] && [ -f "docs/Makefile" ]; then \
		$(MAKE) -C docs livehtml; \
	else \
		@echo "Error: docs directory or docs/Makefile not found."; \
		exit 1; \
	fi 