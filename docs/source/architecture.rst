Architecture
============

PlotSmith follows a strict 4-layer architecture with clear boundaries between layers. This design ensures separation of concerns, testability, and maintainability.

Layer 1: Data and Representations
---------------------------------

**Location**: ``plotsmith.objects``

This layer defines immutable dataclasses that represent plot-ready data and plot specifications. No matplotlib imports are allowed in this layer.

**View Objects**:

- :class:`plotsmith.objects.SeriesView` - Time series data
- :class:`plotsmith.objects.BandView` - Confidence bands
- :class:`plotsmith.objects.ScatterView` - Scatter plot data
- :class:`plotsmith.objects.HistogramView` - Histogram data
- :class:`plotsmith.objects.BarView` - Bar chart data
- :class:`plotsmith.objects.HeatmapView` - Heatmap data
- :class:`plotsmith.objects.FigureSpec` - Figure specifications

**Validators**: ``plotsmith.objects.validate`` contains validation functions that check shapes and alignment of view objects.

Layer 2: Primitives
--------------------

**Location**: ``plotsmith.primitives``

This layer defines the drawing API. It can import matplotlib and must accept only Layer 1 objects. All matplotlib calls are encapsulated here.

**Drawing Functions**:

- :func:`plotsmith.primitives.draw_series` - Draw time series
- :func:`plotsmith.primitives.draw_band` - Draw confidence bands
- :func:`plotsmith.primitives.draw_scatter` - Draw scatter plots
- :func:`plotsmith.primitives.draw_histogram` - Draw histograms
- :func:`plotsmith.primitives.draw_bar` - Draw bar charts
- :func:`plotsmith.primitives.draw_heatmap` - Draw heatmaps

**Styling Helpers**:

- :func:`plotsmith.primitives.minimal_axes` - Apply minimalist axes styling
- :func:`plotsmith.primitives.tidy_axes` - Tidy axes styling
- :func:`plotsmith.primitives.style_line_plot` - Style line plots
- :func:`plotsmith.primitives.style_scatter_plot` - Style scatter plots
- :func:`plotsmith.primitives.style_bar_plot` - Style bar plots

**Labeling Helpers**:

- :func:`plotsmith.primitives.direct_label` - Add direct labels
- :func:`plotsmith.primitives.note` - Add notes
- :func:`plotsmith.primitives.emphasize_last` - Emphasize last point
- :func:`plotsmith.primitives.accent_point` - Highlight specific points
- :func:`plotsmith.primitives.event_line` - Mark events with lines

Layer 3: Tasks
--------------

**Location**: ``plotsmith.tasks``

Tasks translate user intent into Layer 1 objects and a FigureSpec. Tasks can import pandas and numpy, but not matplotlib. Tasks can only import Layer 1 objects.

**Task Classes**:

- :class:`plotsmith.tasks.TimeseriesPlotTask` - Convert time series data to views
- :class:`plotsmith.tasks.BacktestPlotTask` - Convert backtest results to views
- :class:`plotsmith.tasks.HistogramPlotTask` - Convert data to histogram views
- :class:`plotsmith.tasks.BarPlotTask` - Convert data to bar views
- :class:`plotsmith.tasks.HeatmapPlotTask` - Convert data to heatmap views
- :class:`plotsmith.tasks.ResidualsPlotTask` - Convert residuals to views

Layer 4: Workflows
------------------

**Location**: ``plotsmith.workflows``

Workflows provide one-call entry points for users. Workflows can import matplotlib. They call tasks, then primitives, then save or show plots.

**Workflow Functions**:

- :func:`plotsmith.workflows.plot_timeseries` - Plot time series
- :func:`plotsmith.workflows.plot_backtest` - Plot backtest results
- :func:`plotsmith.workflows.plot_residuals` - Plot residuals
- :func:`plotsmith.workflows.plot_histogram` - Plot histograms
- :func:`plotsmith.workflows.plot_bar` - Plot bar charts
- :func:`plotsmith.workflows.plot_heatmap` - Plot heatmaps
- :func:`plotsmith.workflows.plot_model_comparison` - Compare models
- :func:`plotsmith.workflows.figure` - Create styled figure
- :func:`plotsmith.workflows.small_multiples` - Create small multiples

Public API
----------

The public API is defined in ``plotsmith/__init__.py`` and exports only:

- Workflow functions
- View objects (for type hints)
- Styling helpers

All other modules are internal and should not be imported directly.


