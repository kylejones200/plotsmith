# Local Development Setup

To catch linting errors locally before pushing to CI, follow these steps:

## 1. Install Dev Dependencies

```bash
pip install -e ".[dev]"
```

This installs:
- `ruff` (linter and formatter)
- `mypy` (type checker)
- `pytest` (testing framework)
- `pytest-cov` (coverage)
- `pytest-benchmark` (benchmarks)

## 2. Set Up Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This will automatically run linting and formatting checks before each commit.

## 3. Manual Linting (Optional)

You can also run linting manually:

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

## 4. Run Tests Locally

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=plotsmith --cov-report=html

# Run specific test file
pytest tests/test_workflows.py
```

## Why Errors Weren't Caught Locally

If you see linting errors in CI but not locally, it's usually because:

1. **Dev dependencies not installed**: Run `pip install -e ".[dev]"`
2. **Pre-commit hooks not installed**: Run `pre-commit install`
3. **Not running linter manually**: Use `ruff check` before committing

The CI always installs dev dependencies and runs the linter, so errors will show up there even if your local setup is missing them.

