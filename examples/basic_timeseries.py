"""Basic timeseries plotting example.

This example demonstrates plotsmith working with timesmith.typing:
1. Creates a pandas Series (SeriesLike)
2. Validates via timesmith.typing
3. Plots the series
"""

import numpy as np
import pandas as pd
from pathlib import Path

from plotsmith import plot_timeseries
from timesmith.typing import SeriesLike
from timesmith.typing.validators import assert_series_like

# Set deterministic seed
np.random.seed(42)

# Create synthetic time series data
dates = pd.date_range("2020-01-01", periods=100, freq="D")
trend = np.linspace(0, 10, 100)
noise = np.random.normal(0, 1, 100)
values = trend + noise

series: SeriesLike = pd.Series(values, index=dates, name="Synthetic Series")

# Validate via timesmith.typing
assert_series_like(series)
print("✓ Validated SeriesLike via timesmith.typing")

# Create confidence bands
lower = pd.Series(values - 2, index=dates)
upper = pd.Series(values + 2, index=dates)
bands = {"95% Confidence": (lower, upper)}

# Create output directory if it doesn't exist
output_dir = Path(__file__).parent / "out"
output_dir.mkdir(exist_ok=True)

# Plot the timeseries
fig, ax = plot_timeseries(
    data=series,
    bands=bands,
    title="Synthetic Time Series with Confidence Bands",
    xlabel="Date",
    ylabel="Value",
    save_path=output_dir / "basic_timeseries.png",
)

print(f"Plot saved to {output_dir / 'basic_timeseries.png'}")

