"""Tasks that convert user data into Layer 1 view objects."""

from typing import Optional

import numpy as np
import pandas as pd

from plotsmith.objects.views import (
    BandView,
    BarView,
    FigureSpec,
    HeatmapView,
    HistogramView,
    ScatterView,
    SeriesView,
)


class TimeseriesPlotTask:
    """Task to create views for time series plotting.

    Accepts pandas Series or DataFrame and produces SeriesView objects
    plus a FigureSpec.
    """

    def __init__(
        self,
        data: pd.Series | pd.DataFrame,
        bands: Optional[dict[str, tuple[pd.Series, pd.Series]]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ):
        """Initialize the timeseries plot task.

        Args:
            data: Time series data as pandas Series or DataFrame.
            bands: Optional dict mapping band names to (lower, upper) Series tuples.
            title: Optional plot title.
            xlabel: Optional X-axis label.
            ylabel: Optional Y-axis label.
        """
        self.data = data
        self.bands = bands or {}
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def execute(self) -> tuple[list[SeriesView | BandView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of views, FigureSpec).
        """
        views: list[SeriesView | BandView] = []

        # Convert data to views
        if isinstance(self.data, pd.Series):
            x = self.data.index.values
            y = self.data.values
            views.append(SeriesView(x=x, y=y, label=self.data.name))
        elif isinstance(self.data, pd.DataFrame):
            x = self.data.index.values
            for col in self.data.columns:
                y = self.data[col].values
                views.append(SeriesView(x=x, y=y, label=col))
        else:
            raise TypeError(f"data must be pandas Series or DataFrame, got {type(self.data)}")

        # Add bands if provided
        for band_name, (lower, upper) in self.bands.items():
            if not isinstance(lower, pd.Series) or not isinstance(upper, pd.Series):
                raise TypeError("band values must be pandas Series")
            if not lower.index.equals(upper.index):
                raise ValueError("band lower and upper must have matching indices")
            if not lower.index.equals(self.data.index):
                raise ValueError("band indices must match data indices")

            x_band = lower.index.values
            y_lower = lower.values
            y_upper = upper.values
            views.append(BandView(x=x_band, y_lower=y_lower, y_upper=y_upper, label=band_name))

        # Create figure spec
        spec = FigureSpec(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        return views, spec


class BacktestPlotTask:
    """Task to create views for backtest results plotting.

    Accepts a results table with y_true and y_pred columns plus fold_id.
    Produces scatter views for predictions vs actuals.
    """

    def __init__(
        self,
        results: pd.DataFrame,
        y_true_col: str = "y_true",
        y_pred_col: str = "y_pred",
        fold_id_col: Optional[str] = "fold_id",
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ):
        """Initialize the backtest plot task.

        Args:
            results: DataFrame with y_true, y_pred, and optionally fold_id columns.
            y_true_col: Name of the true values column.
            y_pred_col: Name of the predicted values column.
            fold_id_col: Optional name of the fold ID column for grouping.
            title: Optional plot title.
            xlabel: Optional X-axis label.
            ylabel: Optional Y-axis label.
        """
        self.results = results
        self.y_true_col = y_true_col
        self.y_pred_col = y_pred_col
        self.fold_id_col = fold_id_col
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def execute(self) -> tuple[list[ScatterView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of ScatterView objects, FigureSpec).
        """
        if self.y_true_col not in self.results.columns:
            raise ValueError(f"Column '{self.y_true_col}' not found in results")
        if self.y_pred_col not in self.results.columns:
            raise ValueError(f"Column '{self.y_pred_col}' not found in results")

        views: list[ScatterView] = []

        if self.fold_id_col is not None and self.fold_id_col in self.results.columns:
            # Group by fold_id and create separate views for each fold
            for fold_id, group in self.results.groupby(self.fold_id_col):
                x = group[self.y_true_col].values
                y = group[self.y_pred_col].values
                views.append(ScatterView(x=x, y=y, label=f"Fold {fold_id}"))
        else:
            # Single scatter view for all data
            x = self.results[self.y_true_col].values
            y = self.results[self.y_pred_col].values
            views.append(ScatterView(x=x, y=y))

        # Create figure spec
        spec = FigureSpec(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        return views, spec


class HistogramPlotTask:
    """Task to create views for histogram plotting.

    Accepts array-like data and produces HistogramView objects plus a FigureSpec.
    """

    def __init__(
        self,
        data: np.ndarray | pd.Series,
        bins: Optional[int | np.ndarray] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        label: Optional[str] = None,
        color: Optional[str] = None,
        edgecolor: Optional[str] = None,
        alpha: Optional[float] = None,
    ):
        """Initialize the histogram plot task.

        Args:
            data: Data to histogram (array-like or pandas Series).
            bins: Optional number of bins or bin edges.
            title: Optional plot title.
            xlabel: Optional X-axis label.
            ylabel: Optional Y-axis label.
            label: Optional label for the histogram.
            color: Optional fill color.
            edgecolor: Optional edge color.
            alpha: Optional transparency.
        """
        self.data = np.asarray(data) if not isinstance(data, np.ndarray) else data
        self.bins = bins
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.label = label
        self.color = color
        self.edgecolor = edgecolor
        self.alpha = alpha

    def execute(self) -> tuple[list[HistogramView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of HistogramView objects, FigureSpec).
        """
        view = HistogramView(
            values=self.data,
            bins=self.bins,
            label=self.label,
            color=self.color,
            edgecolor=self.edgecolor,
            alpha=self.alpha,
        )

        spec = FigureSpec(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        return [view], spec


class BarPlotTask:
    """Task to create views for bar chart plotting.

    Accepts categories and values and produces BarView objects plus a FigureSpec.
    """

    def __init__(
        self,
        x: np.ndarray | list[str] | pd.Series | pd.Index,
        height: np.ndarray | pd.Series,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        label: Optional[str] = None,
        color: Optional[str] = None,
        edgecolor: Optional[str] = None,
        alpha: Optional[float] = None,
        horizontal: bool = False,
    ):
        """Initialize the bar plot task.

        Args:
            x: X-axis positions (categories or positions).
            height: Bar heights.
            title: Optional plot title.
            xlabel: Optional X-axis label.
            ylabel: Optional Y-axis label.
            label: Optional label for the bars.
            color: Optional fill color.
            edgecolor: Optional edge color.
            alpha: Optional transparency.
            horizontal: If True, create horizontal bars.
        """
        # Convert to appropriate types
        if isinstance(x, (pd.Series, pd.Index)):
            self.x = x.tolist() if isinstance(x, pd.Index) else x.values
        elif isinstance(x, list):
            self.x = x
        else:
            self.x = np.asarray(x)

        self.height = np.asarray(height) if not isinstance(height, np.ndarray) else height
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.label = label
        self.color = color
        self.edgecolor = edgecolor
        self.alpha = alpha
        self.horizontal = horizontal

    def execute(self) -> tuple[list[BarView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of BarView objects, FigureSpec).
        """
        view = BarView(
            x=self.x,
            height=self.height,
            label=self.label,
            color=self.color,
            edgecolor=self.edgecolor,
            alpha=self.alpha,
            horizontal=self.horizontal,
        )

        spec = FigureSpec(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        return [view], spec


class HeatmapPlotTask:
    """Task to create views for heatmap plotting.

    Accepts 2D array or DataFrame and produces HeatmapView objects plus a FigureSpec.
    """

    def __init__(
        self,
        data: np.ndarray | pd.DataFrame,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        annotate: bool = False,
        fmt: Optional[str] = None,
    ):
        """Initialize the heatmap plot task.

        Args:
            data: 2D array or DataFrame to display as heatmap.
            title: Optional plot title.
            xlabel: Optional X-axis label.
            ylabel: Optional Y-axis label.
            cmap: Optional colormap name.
            vmin: Optional minimum value for color scale.
            vmax: Optional maximum value for color scale.
            annotate: If True, annotate cells with values.
            fmt: Optional format string for annotations.
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.x_labels = data.columns.tolist()
            self.y_labels = data.index.tolist()
        else:
            self.data = np.asarray(data)
            self.x_labels = None
            self.y_labels = None

        if self.data.ndim != 2:
            raise ValueError(f"Data must be 2D, got {self.data.ndim}D")

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.annotate = annotate
        self.fmt = fmt

    def execute(self) -> tuple[list[HeatmapView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of HeatmapView objects, FigureSpec).
        """
        view = HeatmapView(
            data=self.data,
            x_labels=self.x_labels,
            y_labels=self.y_labels,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            annotate=self.annotate,
            fmt=self.fmt,
        )

        spec = FigureSpec(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        return [view], spec


class ResidualsPlotTask:
    """Task to create views for residual plotting.

    Accepts actual and predicted values and produces ScatterView or SeriesView objects
    for residual analysis.
    """

    def __init__(
        self,
        actual: np.ndarray | pd.Series,
        predicted: np.ndarray | pd.Series,
        x: Optional[np.ndarray | pd.Series] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        plot_type: str = "scatter",  # "scatter" or "series"
    ):
        """Initialize the residuals plot task.

        Args:
            actual: Actual/true values.
            predicted: Predicted values.
            x: Optional x-axis values (for series plot). If None, uses index.
            title: Optional plot title.
            xlabel: Optional X-axis label.
            ylabel: Optional Y-axis label.
            plot_type: Type of plot - "scatter" for scatter plot, "series" for time series.
        """
        self.actual = np.asarray(actual) if not isinstance(actual, np.ndarray) else actual
        self.predicted = (
            np.asarray(predicted) if not isinstance(predicted, np.ndarray) else predicted
        )

        if len(self.actual) != len(self.predicted):
            raise ValueError(
                f"actual and predicted must have same length, got {len(self.actual)} and {len(self.predicted)}"
            )

        self.residuals = self.actual - self.predicted

        if x is not None:
            self.x = np.asarray(x) if not isinstance(x, np.ndarray) else x
        elif isinstance(actual, pd.Series):
            self.x = actual.index.values
        else:
            self.x = np.arange(len(self.actual))

        if len(self.x) != len(self.actual):
            raise ValueError(
                f"x must have same length as actual, got {len(self.x)} and {len(self.actual)}"
            )

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type

    def execute(self) -> tuple[list[ScatterView | SeriesView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of view objects, FigureSpec).
        """
        views: list[ScatterView | SeriesView] = []

        if self.plot_type == "scatter":
            # Scatter plot: predicted vs actual
            views.append(ScatterView(x=self.actual, y=self.predicted, label="Residuals"))
        elif self.plot_type == "series":
            # Time series plot: residuals over time
            views.append(SeriesView(x=self.x, y=self.residuals, label="Residuals"))
        else:
            raise ValueError(f"plot_type must be 'scatter' or 'series', got {self.plot_type}")

        spec = FigureSpec(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        return views, spec

