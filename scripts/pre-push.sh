#!/bin/bash
# Pre-push hook to run linting and formatting checks
# This ensures code quality before pushing to remote

set -e

echo "Running pre-push checks..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if ruff is available
if ! command -v ruff &> /dev/null; then
    echo "Error: ruff not found. Install dev dependencies: pip install -e '.[dev]'"
    exit 1
fi

# Run formatting check
echo "Checking code formatting..."
if ! ruff format --check plotsmith/ tests/; then
    echo "Error: Code formatting issues found."
    echo "Run 'ruff format plotsmith/ tests/' to fix them."
    exit 1
fi

# Run linting check
echo "Checking code quality..."
if ! ruff check plotsmith/ tests/; then
    echo "Error: Linting issues found."
    echo "Run 'ruff check --fix plotsmith/ tests/' to fix them."
    exit 1
fi

# Run type checking (optional, can be slow)
echo "Running type checks..."
if command -v mypy &> /dev/null; then
    if ! mypy plotsmith/; then
        echo "Warning: Type checking issues found (non-blocking)."
    fi
fi

echo "All pre-push checks passed!"
exit 0

