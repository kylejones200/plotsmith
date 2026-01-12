#!/usr/bin/env python3
"""Smoke test for plotsmith integration with timesmith.

This script validates that plotsmith works correctly with timesmith typing.
It must:
1. Accept a pandas Series
2. Validate via timesmith.typing
3. Plot the series
4. Exit 0 on success
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

import numpy as np
import pandas as pd

# Import from timesmith.typing (single source of truth)
from timesmith.typing import SeriesLike
from timesmith.typing.validators import assert_series_like

# Import plotsmith
from plotsmith import plot_timeseries


def main() -> int:
    """Run smoke test."""
    # Create a pandas Series
    np.random.seed(42)
    n = 50
    values = np.random.randn(n) * 2 + 10
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    y: SeriesLike = pd.Series(values, index=index, name="Test Series")
    
    # Validate via timesmith.typing
    try:
        assert_series_like(y)
        print("✓ Validated SeriesLike via timesmith.typing")
    except Exception as e:
        print(f"ERROR: timesmith validation failed: {e}", file=sys.stderr)
        return 1
    
    # Use plotsmith to plot
    try:
        output_dir = Path(__file__).parent.parent / "examples" / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plot_timeseries(
            data=y,
            title="Smoke Test Plot",
            xlabel="Date",
            ylabel="Value",
            save_path=output_dir / "smoke_test.png",
        )
        
        print("✓ Created plot successfully")
        print(f"✓ Smoke test passed: plotsmith works with timesmith typing")
        return 0
        
    except Exception as e:
        print(f"ERROR: plotsmith plotting failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

