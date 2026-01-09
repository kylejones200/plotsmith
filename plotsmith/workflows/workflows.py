"""Workflows that orchestrate tasks and primitives."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotsmith.objects.validate import validate_all_views
from plotsmith.objects.views import BarView, FigureSpec, HeatmapView, HistogramView, SeriesView
from plotsmith.primitives.draw import (
    apply_axes_style,
    draw_band,
    draw_bar,
    draw_heatmap,
    draw_histogram,
    draw_scatter,
    draw_series,
    event_line,
    force_bar_zero,
    minimal_axes,
)
from plotsmith.tasks.tasks import (
    BacktestPlotTask,
    BarPlotTask,
    HeatmapPlotTask,
    HistogramPlotTask,
    ResidualsPlotTask,
    TimeseriesPlotTask,
)

logger = logging.getLogger(__name__)


def plot_timeseries(
    data: pd.Series | pd.DataFrame,
    bands: Optional[dict[str, tuple[pd.Series, pd.Series]]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a time series with optional confidence bands.

    Args:
        data: Time series data as pandas Series or DataFrame.
        bands: Optional dict mapping band names to (lower, upper) Series tuples.
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_timeseries workflow")
    try:
        # Create task and execute
        task = TimeseriesPlotTask(data=data, bands=bands, title=title, xlabel=xlabel, ylabel=ylabel)
        views, spec = task.execute()

        # Validate views
        validate_all_views(views)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Apply minimal styling
        minimal_axes(ax)

        # Draw all views
        for view in views:
            if isinstance(view, HeatmapView):
                draw_heatmap(ax, view)
            elif isinstance(view, HistogramView):
                draw_histogram(ax, view)
            elif isinstance(view, BarView):
                draw_bar(ax, view)
            elif hasattr(view, "y_lower"):  # BandView
                draw_band(ax, view)
            elif hasattr(view, "s"):  # ScatterView
                draw_scatter(ax, view)
            else:  # SeriesView
                draw_series(ax, view)

        # Apply figure spec
        apply_axes_style(ax, spec)

        # Add legend if any views have labels
        if any(getattr(v, "label", None) for v in views):
            ax.legend()

        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_timeseries workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_timeseries workflow: {e}")
        raise


def plot_backtest(
    results: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    fold_id_col: Optional[str] = "fold_id",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (8, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot backtest results as scatter plot of predictions vs actuals.

    Args:
        results: DataFrame with y_true, y_pred, and optionally fold_id columns.
        y_true_col: Name of the true values column.
        y_pred_col: Name of the predicted values column.
        fold_id_col: Optional name of the fold ID column for grouping.
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_backtest workflow")
    try:
        # Create task and execute
        task = BacktestPlotTask(
            results=results,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            fold_id_col=fold_id_col,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        views, spec = task.execute()

        # Validate views
        validate_all_views(views)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Apply minimal styling
        minimal_axes(ax)

        # Draw all views
        for view in views:
            draw_scatter(ax, view)

        # Apply figure spec
        apply_axes_style(ax, spec)

        # Add legend if any views have labels
        if any(view.label for view in views):
            ax.legend()

        # Add diagonal line for perfect predictions
        if len(views) > 0:
            all_x = np.concatenate([v.x for v in views])
            all_y = np.concatenate([v.y for v in views])
            min_val = min(all_x.min(), all_y.min())
            max_val = max(all_x.max(), all_y.max())
            ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Perfect")

        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_backtest workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_backtest workflow: {e}")
        raise


def plot_histogram(
    data: np.ndarray | pd.Series | list[np.ndarray | pd.Series],
    bins: Optional[int | np.ndarray] = None,
    labels: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    edgecolor: Optional[str] = None,
    alpha: Optional[float] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a histogram (supports single or multiple overlaid histograms).

    Args:
        data: Data to histogram (array-like, pandas Series, or list of arrays for overlaid).
        bins: Optional number of bins or bin edges.
        labels: Optional list of labels for multiple histograms.
        colors: Optional list of colors for multiple histograms.
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        label: Optional label for single histogram (deprecated if using labels).
        color: Optional fill color for single histogram (deprecated if using colors).
        edgecolor: Optional edge color.
        alpha: Optional transparency.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_histogram workflow")
    try:
        # Handle multiple histograms
        if isinstance(data, list):
            views: list[HistogramView] = []
            for i, d in enumerate(data):
                d_array = np.asarray(d) if not isinstance(d, np.ndarray) else d
                label_val = labels[i] if labels and i < len(labels) else f"Series {i+1}"
                color_val = colors[i] if colors and i < len(colors) else None
                views.append(
                    HistogramView(
                        values=d_array,
                        bins=bins,
                        label=label_val,
                        color=color_val,
                        edgecolor=edgecolor,
                        alpha=alpha,
                    )
                )
            spec = FigureSpec(title=title, xlabel=xlabel, ylabel=ylabel)
        else:
            # Single histogram
            task = HistogramPlotTask(
                data=data,
                bins=bins,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                label=label,
                color=color,
                edgecolor=edgecolor,
                alpha=alpha,
            )
            views, spec = task.execute()

        # Validate views
        validate_all_views(views)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Apply minimal styling
        minimal_axes(ax)

        # Draw all views
        for view in views:
            draw_histogram(ax, view)

        # Apply figure spec
        apply_axes_style(ax, spec)

        # Add legend if any views have labels
        if any(view.label for view in views):
            ax.legend()

        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_histogram workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_histogram workflow: {e}")
        raise


def plot_bar(
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
    force_zero: bool = True,
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a bar chart.

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
        force_zero: If True, force y-axis to start at zero (honest scale).
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_bar workflow")
    try:
        # Create task and execute
        task = BarPlotTask(
            x=x,
            height=height,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            label=label,
            color=color,
            edgecolor=edgecolor,
            alpha=alpha,
            horizontal=horizontal,
        )
        views, spec = task.execute()

        # Validate views
        validate_all_views(views)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Apply minimal styling
        minimal_axes(ax)

        # Draw all views
        for view in views:
            draw_bar(ax, view)

        # Force zero if requested
        if force_zero:
            force_bar_zero(ax)

        # Apply figure spec
        apply_axes_style(ax, spec)

        # Add legend if any views have labels
        if any(view.label for view in views):
            ax.legend()

        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_bar workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_bar workflow: {e}")
        raise


def figure(
    *,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    sharex: bool = False,
    sharey: bool = False,
    constrained_layout: bool = True,
) -> tuple[plt.Figure, plt.Axes | list[plt.Axes]]:
    """Create a PlotSmith-styled figure.

    This is a light wrapper around matplotlib.pyplot.subplots that
    chooses sensible defaults for analytical figures and returns the figure
    plus axes.

    Args:
        nrows: Number of rows of subplots.
        ncols: Number of columns of subplots.
        figsize: Figure size tuple (width, height).
        sharex: If True, share x-axis across subplots.
        sharey: If True, share y-axis across subplots.
        constrained_layout: If True, use constrained layout.

    Returns:
        Tuple of (matplotlib Figure, Axes or list of Axes).
    """
    logger.info("Creating figure")
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        constrained_layout=constrained_layout,
    )
    return fig, axes


def small_multiples(
    count: int,
    *,
    ncols: int = 3,
    sharex: bool = True,
    sharey: bool = False,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create a grid of small-multiple axes.

    Returns the figure and a flat sequence of axes, already styled with
    tidy_axes.

    Args:
        count: Number of subplots to create.
        ncols: Number of columns in the grid.
        sharex: If True, share x-axis across subplots.
        sharey: If True, share y-axis across subplots.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, list of Axes).
    """
    logger.info(f"Creating small multiples grid with {count} plots")
    from plotsmith.primitives.draw import tidy_axes

    nrows = (count + ncols - 1) // ncols
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        constrained_layout=True,
    )

    # Flatten axes grid
    if isinstance(axes_grid, plt.Axes):
        axes_list = [axes_grid]
    else:
        axes_list = []
        for row in axes_grid:
            if isinstance(row, np.ndarray):
                axes_list.extend(row)
            else:
                axes_list.append(row)

    # Only keep as many axes as requested and tidy them
    axes_list = axes_list[:count]
    for ax in axes_list:
        tidy_axes(ax)

    return fig, axes_list


def plot_heatmap(
    data: np.ndarray | pd.DataFrame,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate: bool = False,
    fmt: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a heatmap (correlation matrix or 2D data).

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
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_heatmap workflow")
    try:
        # Create task and execute
        task = HeatmapPlotTask(
            data=data,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annotate=annotate,
            fmt=fmt,
        )
        views, spec = task.execute()

        # Validate views
        validate_all_views(views)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Apply minimal styling
        minimal_axes(ax)

        # Draw all views
        for view in views:
            draw_heatmap(ax, view)

        # Apply figure spec
        apply_axes_style(ax, spec)

        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_heatmap workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_heatmap workflow: {e}")
        raise


def plot_model_comparison(
    data: pd.Series | pd.DataFrame,
    models: dict[str, np.ndarray | pd.Series],
    test_start_idx: Optional[int] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (12, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple model predictions compared to actual data.

    Args:
        data: Actual time series data as pandas Series or DataFrame.
        models: Dictionary mapping model names to their predictions (arrays/Series).
        test_start_idx: Optional index where test period starts (for visual separation).
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_model_comparison workflow")
    try:
        # Convert data to Series if DataFrame
        if isinstance(data, pd.DataFrame):
            if len(data.columns) == 1:
                data = data.iloc[:, 0]
            else:
                raise ValueError("DataFrame must have single column, or use Series")

        x = data.index.values if isinstance(data, pd.Series) else np.arange(len(data))
        y_actual = data.values

        # Create views
        views: list[SeriesView] = []
        views.append(SeriesView(x=x, y=y_actual, label="Actual", linewidth=2))

        # Add model predictions
        for name, pred in models.items():
            pred_array = np.asarray(pred) if not isinstance(pred, np.ndarray) else pred
            if len(pred_array) != len(y_actual):
                # Align to end if different length
                pred_array = pred_array[-len(y_actual):]
            views.append(SeriesView(x=x, y=pred_array, label=name, linewidth=1.5, alpha=0.7))

        # Validate views
        validate_all_views(views)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Apply minimal styling
        minimal_axes(ax)

        # Draw all views
        for view in views:
            draw_series(ax, view)

        # Add vertical line for test start if provided
        if test_start_idx is not None and test_start_idx < len(x):
            event_line(ax, x[test_start_idx], text="Test Start", color="black", linestyle="--")

        # Create figure spec
        spec = FigureSpec(title=title, xlabel=xlabel, ylabel=ylabel)
        apply_axes_style(ax, spec)

        # Add legend
        ax.legend()

        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_model_comparison workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_model_comparison workflow: {e}")
        raise


def plot_residuals(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    x: Optional[np.ndarray | pd.Series] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    plot_type: str = "scatter",
    add_perfect_line: bool = True,
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot residuals (actual vs predicted or residuals over time).

    Args:
        actual: Actual/true values.
        predicted: Predicted values.
        x: Optional x-axis values (for series plot). If None, uses index.
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        plot_type: Type of plot - "scatter" for scatter plot, "series" for time series.
        add_perfect_line: If True and plot_type is "scatter", add diagonal perfect prediction line.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_residuals workflow")
    try:
        # Create task and execute
        task = ResidualsPlotTask(
            actual=actual,
            predicted=predicted,
            x=x,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_type=plot_type,
        )
        views, spec = task.execute()

        # Validate views
        validate_all_views(views)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Apply minimal styling
        minimal_axes(ax)

        # Draw all views
        for view in views:
            if hasattr(view, "s"):  # ScatterView
                draw_scatter(ax, view)
            else:  # SeriesView
                draw_series(ax, view)

        # Add perfect prediction line for scatter plots
        if plot_type == "scatter" and add_perfect_line and len(views) > 0:
            view = views[0]
            if hasattr(view, "x") and hasattr(view, "y"):
                all_x = view.x
                all_y = view.y
                min_val = min(all_x.min(), all_y.min())
                max_val = max(all_x.max(), all_y.max())
                ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Perfect")

        # Apply figure spec
        apply_axes_style(ax, spec)

        # Add legend if any views have labels
        if any(getattr(v, "label", None) for v in views) or (plot_type == "scatter" and add_perfect_line):
            ax.legend()

        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_residuals workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_residuals workflow: {e}")
        raise

