#!/bin/bash
# Comprehensive CI test script that matches tox exactly
# This script ensures we catch all tox failures before committing

set -e  # Exit on any error

echo "🔍 Running comprehensive CI tests (matching tox exactly)..."
echo ""

# 1. Run tests first
echo "1️⃣ Running tests..."
python3 -m pytest tests/ --tb=short -q
echo "✅ Tests passed"
echo ""

# 2. Check Black formatting (exact tox command)
echo "2️⃣ Checking Black formatting..."
black --check --diff --line-length=120 src/icvision tests/
echo "✅ Black formatting passed"
echo ""

# 3. Check isort import sorting (exact tox command)
echo "3️⃣ Checking isort import sorting..."
isort --check-only --diff --line-length=120 src/icvision tests/
echo "✅ isort import sorting passed"
echo ""

# 4. Check flake8 linting (exact tox command)
echo "4️⃣ Checking flake8 linting..."
flake8 src/icvision tests/
echo "✅ flake8 linting passed"
echo ""

# 5. Check mypy type checking (exact tox command)
echo "5️⃣ Checking mypy type checking..."
mypy --ignore-missing-imports --no-strict-optional --follow-imports=skip src/icvision/
echo "✅ mypy type checking passed"
echo ""

echo "🎉 All CI tests passed! Safe to commit."