# Development Setup Guide

## Why Linting Errors Weren't Caught Locally

The linting errors appeared in CI but not locally because:

1. **Dev dependencies not installed**: `ruff` and `mypy` are in `[dev]` dependencies but weren't installed locally
2. **Pre-commit hooks not set up**: The `.pre-commit-config.yaml` exists but hooks weren't installed
3. **No manual linting**: The linter wasn't run before committing

## Quick Setup (Recommended)

### 1. Create a Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 2. Install Dev Dependencies

```bash
pip install -e ".[dev]"
```

This installs:
- `ruff` (linter and formatter)
- `mypy` (type checker)
- `pytest` (testing framework)
- `pytest-cov` (coverage)
- `pytest-benchmark` (benchmarks)

### 3. Set Up Pre-commit Hooks (Recommended)

```bash
pip install pre-commit
pre-commit install
```

This automatically runs linting and formatting checks before each commit.

### 4. Verify Setup

```bash
# Check ruff is installed
ruff --version

# Check mypy is installed
mypy --version

# Run linting manually
ruff check plotsmith/ tests/

# Run formatting check
ruff format --check plotsmith/ tests/
```

## Manual Linting Commands

If you prefer to run linting manually (without pre-commit hooks):

```bash
# Check for linting errors
ruff check plotsmith/ tests/

# Auto-fix linting errors
ruff check --fix plotsmith/ tests/

# Check formatting
ruff format --check plotsmith/ tests/

# Auto-format code
ruff format plotsmith/ tests/

# Type checking
mypy plotsmith/
```

## Running Tests Locally

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=plotsmith --cov-report=html

# Run specific test file
pytest tests/test_workflows.py

# Run with verbose output
pytest -v
```

## What CI Does

The CI workflow (`.github/workflows/ci.yml`) automatically:

1. Installs dev dependencies: `pip install -e ".[dev]"`
2. Runs linting: `ruff check plotsmith/ tests/`
3. Checks formatting: `ruff format --check plotsmith/ tests/`
4. Runs type checking: `mypy plotsmith/`
5. Runs tests with coverage: `pytest --cov=plotsmith`

If your local setup matches CI, you'll catch errors before pushing.

## Troubleshooting

### "ruff: command not found"
- Make sure you've activated your virtual environment
- Run `pip install -e ".[dev]"` to install ruff

### "pre-commit: command not found"
- Install pre-commit: `pip install pre-commit`
- Then run: `pre-commit install`

### PEP 668 errors (macOS Homebrew Python)
- Use a virtual environment instead of installing globally
- Create venv: `python3 -m venv venv`
- Activate: `source venv/bin/activate`

## Best Practices

1. **Always use a virtual environment** for Python projects
2. **Install dev dependencies** before developing: `pip install -e ".[dev]"`
3. **Set up pre-commit hooks** to catch errors automatically
4. **Run tests locally** before pushing: `pytest`
5. **Check linting** before committing: `ruff check .`

