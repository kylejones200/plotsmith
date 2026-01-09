"""Basic timeseries plotting example."""

import numpy as np
import pandas as pd
from pathlib import Path

from plotsmith import plot_timeseries

# Set deterministic seed
np.random.seed(42)

# Create synthetic time series data
dates = pd.date_range("2020-01-01", periods=100, freq="D")
trend = np.linspace(0, 10, 100)
noise = np.random.normal(0, 1, 100)
values = trend + noise

series = pd.Series(values, index=dates, name="Synthetic Series")

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

