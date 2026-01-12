# PlotSmith

[![PyPI version](https://badge.fury.io/py/plotsmith.svg)](https://badge.fury.io/py/plotsmith)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/kylejones200/plotsmith/workflows/Tests/badge.svg)](https://github.com/kylejones200/plotsmith/actions)
[![Documentation](https://readthedocs.org/projects/plotsmith/badge/?version=latest)](https://plotsmith.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**PlotSmith** is a layered plotting library for ML models with strict architectural boundaries. It provides clean, minimalist visualizations following best practices for analytical figures.

## Features

- **4-Layer Architecture**: Strict separation of concerns with clear boundaries
- **Minimalist Styling**: Clean, publication-ready plots with serif fonts and minimal clutter
- **Type-Safe**: Full type hints and immutable data structures
- **Comprehensive Chart Types**: 
  - **Time Series**: Time series plots with confidence bands
  - **Statistical**: Histograms, box plots, violin plots, scatter plots
  - **Categorical**: Bar charts, lollipop charts, dumbbell charts, range charts
  - **Specialized**: Waterfall charts, waffle charts, slope charts, metric displays
  - **ML Analysis**: Residual plots, backtest visualization, model comparison, forecast comparison
  - **Correlation**: Correlation heatmaps
- **ML-Focused**: Built for model evaluation, backtesting, and analysis workflows

## Installation

```bash
pip install plotsmith
```

## Quick Start

```python
import pandas as pd
import numpy as np
from plotsmith import plot_timeseries, plot_residuals, plot_heatmap

# Time series with confidence bands
data = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
lower = data - 0.5
upper = data + 0.5
fig, ax = plot_timeseries(
    data,
    bands={'95% CI': (lower, upper)},
    title='Time Series with Confidence Bands'
)

# Residual analysis
actual = np.array([1, 2, 3, 4, 5])
predicted = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
fig, ax = plot_residuals(actual, predicted, plot_type='scatter')

# Correlation heatmap
corr_matrix = df.corr()
fig, ax = plot_heatmap(corr_matrix, annotate=True, cmap='RdYlGn')
```

## Architecture

PlotSmith follows a strict 4-layer architecture with clear boundaries:

### Layer 1: Objects (`plotsmith.objects`)
Immutable dataclasses representing plot-ready data. **No matplotlib imports.**

- `SeriesView` - Time series data
- `BandView` - Confidence bands
- `ScatterView` - Scatter plot data
- `HistogramView` - Histogram data
- `BarView` - Bar chart data
- `HeatmapView` - 2D heatmap data
- `FigureSpec` - Figure styling specification

### Layer 2: Primitives (`plotsmith.primitives`)
Drawing functions that accept only Layer 1 objects. **All matplotlib calls here.**

- `draw_series()`, `draw_scatter()`, `draw_histogram()`, `draw_bar()`, `draw_heatmap()`
- `minimal_axes()` - Minimalist axes styling
- `tidy_axes()` - SignalPlot-style axes cleanup
- `style_line_plot()`, `style_scatter_plot()`, `style_bar_plot()`
- `event_line()`, `direct_label()`, `note()`, `accent_point()`

### Layer 3: Tasks (`plotsmith.tasks`)
Translate user intent into Layer 1 objects. **Can import pandas/numpy, not matplotlib.**

- `TimeseriesPlotTask` - Convert Series/DataFrame to views
- `BacktestPlotTask` - Convert results table to scatter views
- `HistogramPlotTask` - Convert data to histogram views
- `BarPlotTask` - Convert data to bar views
- `HeatmapPlotTask` - Convert 2D data to heatmap views
- `ResidualsPlotTask` - Convert actual/predicted to residual views

### Layer 4: Workflows (`plotsmith.workflows`)
Public API entry points. **Orchestrate tasks → primitives → save/show.**

- `plot_timeseries()` - Time series plotting
- `plot_backtest()` - Backtest results visualization
- `plot_histogram()` - Histogram plotting (supports overlaid)
- `plot_bar()` - Bar chart plotting
- `plot_heatmap()` - Heatmap/correlation matrix plotting
- `plot_residuals()` - Residual analysis plots
- `plot_model_comparison()` - Compare multiple model predictions
- `figure()` - Create styled figure
- `small_multiples()` - Create grid of small-multiple axes

## Workflows

### Time Series Plotting

```python
from plotsmith import plot_timeseries
import pandas as pd

# Single series
data = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
fig, ax = plot_timeseries(data, title='My Time Series')

# Multiple series (DataFrame)
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, 
                  index=pd.date_range('2020-01-01', periods=3))
fig, ax = plot_timeseries(df)

# With confidence bands
bands = {'CI': (lower_series, upper_series)}
fig, ax = plot_timeseries(data, bands=bands, save_path='output.png')
```

### Residual Analysis

```python
from plotsmith import plot_residuals

# Scatter plot (predicted vs actual)
fig, ax = plot_residuals(actual, predicted, plot_type='scatter', add_perfect_line=True)

# Time series (residuals over time)
fig, ax = plot_residuals(actual, predicted, x=dates, plot_type='series')
```

### Model Comparison

```python
from plotsmith import plot_model_comparison

models = {
    'LSTM': lstm_predictions,
    'XGBoost': xgb_predictions,
    'SARIMA': sarima_predictions
}
fig, ax = plot_model_comparison(
    actual_data,
    models,
    test_start_idx=100,  # Optional: mark test period start
    title='Model Comparison'
)
```

### Heatmaps

```python
from plotsmith import plot_heatmap
import pandas as pd

# Correlation matrix
corr = df.corr()
fig, ax = plot_heatmap(
    corr,
    annotate=True,
    fmt='.2f',
    cmap='RdYlGn',
    vmin=-1,
    vmax=1
)

# Any 2D array
data = np.random.rand(10, 10)
fig, ax = plot_heatmap(data, title='Custom Heatmap')
```

### Histograms

```python
from plotsmith import plot_histogram

# Single histogram
fig, ax = plot_histogram(data, bins=30, title='Distribution')

# Overlaid histograms
fig, ax = plot_histogram(
    [normal_data, anomaly_data],
    labels=['Normal', 'Anomalies'],
    colors=['blue', 'red'],
    alpha=0.7
)
```

### Bar Charts

```python
from plotsmith import plot_bar

categories = ['A', 'B', 'C', 'D']
values = [10, 20, 15, 25]
fig, ax = plot_bar(
    categories,
    values,
    force_zero=True,  # Honest scale
    title='Bar Chart'
)
```

### Backtest Results

```python
from plotsmith import plot_backtest

results = pd.DataFrame({
    'y_true': [1, 2, 3, 4, 5],
    'y_pred': [1.1, 2.2, 2.9, 4.1, 4.8],
    'fold_id': [0, 0, 1, 1, 2]
})
fig, ax = plot_backtest(results, fold_id_col='fold_id')
```

## Styling Helpers

PlotSmith provides minimalist styling by default and includes helpers for customization:

```python
from plotsmith import minimal_axes, tidy_axes, event_line, direct_label, ACCENT

fig, ax = plt.subplots()
minimal_axes(ax)  # Minimalist styling (serif font, clean spines)
# or
tidy_axes(ax)     # SignalPlot-style (gray spines, clean ticks)

# Add event markers
event_line(ax, x=2020, text='Policy Change', color=ACCENT)

# Direct labels
direct_label(ax, x=5, y=10, text='Peak', use_accent=True)
```

## Examples

See `examples/basic_timeseries.py` for a complete example:

```bash
python examples/basic_timeseries.py
```

This generates `examples/out/basic_timeseries.png` with a time series plot including confidence bands.

## Architecture Principles

1. **Strict Boundaries**: Each layer can only import from layers below it
2. **Immutable Data**: All view objects are frozen dataclasses
3. **Validation**: Shape and alignment checks before drawing
4. **Separation of Concerns**: Data representation (Layer 1) is separate from drawing (Layer 2)
5. **Type Safety**: Full type hints throughout

## Requirements

- Python 3.12+ (only)
- matplotlib >= 3.5.0, < 4.0.0
- numpy >= 1.20.0, < 3.0.0
- pandas >= 1.3.0, < 3.0.0

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=plotsmith --cov-report=html
```

## License

MIT License - see LICENSE file for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use PlotSmith in your research, please cite:

```bibtex
@software{plotsmith,
  title={PlotSmith: A Layered Plotting Library for ML Models},
  author={Jones, Kyle T.},
  year={2025},
  url={https://github.com/kylejones200/plotsmith}
}
```
