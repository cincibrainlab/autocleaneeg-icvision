# Contributing to ICVision

First off, thank you for considering contributing to ICVision! Your help is greatly appreciated. Whether it's reporting a bug, proposing a feature, or writing code, your contributions are valuable.

This document provides guidelines for contributing to ICVision. Please read it carefully to ensure a smooth and effective contribution process.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Pull Requests](#pull-requests)
- [Style Guides](#style-guides)
  - [Python Code Style](#python-code-style)
  - [Commit Messages](#commit-messages)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Contact](#contact)

## Code of Conduct

By participating in this project, you are expected to uphold professional and respectful behavior. Please report unacceptable behavior to [gavin.gammoh@cchmc.org](mailto:gavin.gammoh@cchmc.org) or [nathan.suer@cchmc.org](mailto:nathan.suer@cchmc.org).

## How Can I Contribute?

### Reporting Bugs

Bugs are tracked as [GitHub issues](https://github.com/cincibrainlab/ICVision/issues). If you find a bug, please ensure the bug has not already been reported by searching the issues. If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.

When reporting a bug, please include:

-   **ICVision version** you are using (`icvision --version`).
-   **Python version**.
-   **Operating System** details.
-   A clear and concise **description of the bug**.
-   **Steps to reproduce** the bug.
-   What you **expected to happen**.
-   What **actually happened** (including any error messages and tracebacks).
-   Minimal, reproducible **example code** if applicable.

### Suggesting Enhancements

Enhancement suggestions are tracked as [GitHub issues](https://github.com/cincibrainlab/ICVision/issues).

When suggesting an enhancement, please include:

-   A clear and concise **description of the proposed enhancement**.
-   **Why this enhancement would be useful** to ICVision users.
-   **Specific examples** of how this enhancement would be used.
-   Any **potential drawbacks or considerations**.

### Your First Code Contribution

Unsure where to begin contributing to ICVision? You can start by looking through these `good first issue` and `help wanted` issues:

-   [Good first issues](https://github.com/cincibrainlab/ICVision/labels/good%20first%20issue) - issues which should only require a few lines of code, and a test or two.
-   [Help wanted issues](https://github.com/cincibrainlab/ICVision/labels/help%20wanted) - issues which should be a bit more involved than `good first issues`.

### Pull Requests

When you're ready to contribute code, follow these steps:

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally: `git clone https://github.com/your-username/ICVision.git`
3.  **Create a new branch** for your changes: `git checkout -b feature/your-feature-name` or `bugfix/issue-number`.
4.  **Set up your development environment** (see [Development Setup](#development-setup)).
5.  **Make your changes**. Ensure your code follows the [Style Guides](#style-guides).
6.  **Add tests** for your changes. Ensure all tests pass (see [Testing](#testing)).
7.  **Commit your changes** with a descriptive commit message (see [Commit Messages](#commit-messages)).
8.  **Push your branch** to your fork on GitHub: `git push origin feature/your-feature-name`.
9.  **Open a pull request** to the `main` branch of the `cincibrainlab/ICVision` repository.
    -   Provide a clear title and description for your pull request.
    -   Reference any relevant issues (e.g., "Closes #123").
    -   Ensure all CI checks pass.

## Style Guides

### Python Code Style

-   Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code.
-   Use **Black** for code formatting with 120 character line length. This is enforced by pre-commit hooks.
-   Use **Flake8** for linting. This is also enforced by pre-commit hooks.
-   Use **MyPy** for static type checking.
-   Write clear, concise, and well-commented code. Docstrings should follow [PEP 257](https://www.python.org/dev/peps/pep-0257/), using the Google style for docstrings (as generally used in MNE-Python).

### Commit Messages

-   Use conventional commit messages (e.g., `feat: add new component plotting option`, `fix: resolve API parsing error`).
-   The first line should be a short summary (max 50 characters).
-   If necessary, add a blank line and then a more detailed explanation.
-   Reference issue numbers if applicable (e.g., `fix: correct data loading for .edf files (closes #42)`).

## Development Setup

1.  Clone your fork of the repository.
2.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  Install dependencies, including development tools:
    ```bash
    pip install -e ".[dev,test,docs]"
    ```
4.  Install pre-commit hooks:
    ```bash
    pre-commit install
    ```
    Now, `black`, `flake8`, and other checks will run automatically before each commit.

### Using Makefile (Recommended)

The project includes a Makefile for common development tasks:

```bash
# Install in development mode with all dependencies
make install-dev

# Format code (black + isort)
make format

# Run linters (flake8 + mypy)
make lint

# Run tests
make test

# Run tests with coverage
make coverage
```

## Testing

ICVision uses `pytest` for testing.

### Quick Testing
-   Run all tests:
    ```bash
    pytest
    # OR using Makefile:
    make test
    ```
-   Run tests with coverage report:
    ```bash
    pytest --cov=icvision --cov-report=html
    # OR using Makefile:
    make coverage
    ```
    (Open `htmlcov/index.html` in your browser to view the report.)
-   Run tests for a specific file or directory:
    ```bash
    pytest tests/test_utils.py
    ```

### Comprehensive Testing
-   **IMPORTANT**: Before submitting a PR, run the comprehensive CI test script:
    ```bash
    ./test_ci.sh
    ```
    This runs the same checks as GitHub Actions (formatting, linting, type checking, tests).

-   Run tests with `tox` to check against multiple Python versions:
    ```bash
    tox
    ```

Please ensure that your contributions include appropriate tests and that all existing and new tests pass before submitting a pull request.

## Contact

If you have questions or need to discuss something, you can:
-   Open an issue on GitHub.
-   (If applicable, add other contact methods like a mailing list or chat channel.)

Thank you for contributing to ICVision!
