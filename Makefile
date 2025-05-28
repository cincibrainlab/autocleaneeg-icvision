.PHONY: help install install-dev lint format test coverage clean build publish docs docs-clean docs-live

# Variables
PYTHON = python3
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