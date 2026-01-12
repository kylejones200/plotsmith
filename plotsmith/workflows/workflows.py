"""Workflows that orchestrate tasks and primitives."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotsmith.objects.validate import validate_all_views
from plotsmith.objects.views import (
    BarView,
    FigureSpec,
    HeatmapView,
    HistogramView,
    SeriesView,
)
from plotsmith.primitives.draw import (
    apply_axes_style,
    draw_band,
    draw_bar,
    draw_box,
    draw_dumbbell,
    draw_heatmap,
    draw_histogram,
    draw_lollipop,
    draw_metric,
    draw_range,
    draw_scatter,
    draw_series,
    draw_slope,
    draw_violin,
    draw_waffle,
    draw_waterfall,
    event_line,
    force_bar_zero,
    minimal_axes,
)
from plotsmith.tasks.tasks import (
    BacktestPlotTask,
    BarPlotTask,
    BoxPlotTask,
    CorrelationPlotTask,
    DumbbellPlotTask,
    ForecastComparisonPlotTask,
    HeatmapPlotTask,
    HistogramPlotTask,
    LollipopPlotTask,
    MetricPlotTask,
    RangePlotTask,
    ResidualsPlotTask,
    ScatterPlotTask,
    SlopePlotTask,
    TimeseriesPlotTask,
    ViolinPlotTask,
    WafflePlotTask,
    WaterfallPlotTask,
)

logger = logging.getLogger(__name__)


def plot_timeseries(
    data: pd.Series | pd.DataFrame,
    bands: dict[str, tuple[pd.Series, pd.Series]] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
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
    fold_id_col: str | None = "fold_id",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
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
    bins: int | np.ndarray | None = None,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    label: str | None = None,
    color: str | None = None,
    edgecolor: str | None = None,
    alpha: float | None = None,
    save_path: str | Path | None = None,
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
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    label: str | None = None,
    color: str | None = None,
    edgecolor: str | None = None,
    alpha: float | None = None,
    horizontal: bool = False,
    force_zero: bool = True,
    save_path: str | Path | None = None,
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
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = False,
    fmt: str | None = None,
    save_path: str | Path | None = None,
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
    test_start_idx: int | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
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
    x: np.ndarray | pd.Series | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    plot_type: str = "scatter",
    add_perfect_line: bool = True,
    save_path: str | Path | None = None,
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


def plot_waterfall(
    df: pd.DataFrame,
    categories_col: str,
    values_col: str,
    measure_col: str | None = None,
    colors: list[str] | None = None,
    color: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a waterfall chart.

    Args:
        df: DataFrame with categories and values columns.
        categories_col: Name of the categories column.
        values_col: Name of the values column.
        measure_col: Optional name of the measure type column ('absolute', 'relative', 'total').
        colors: Optional list of colors for each bar.
        color: Optional single color for all bars.
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_waterfall workflow")
    try:
        task = WaterfallPlotTask(
            categories=df[categories_col].tolist(),
            values=df[values_col].values,
            measures=df[measure_col].tolist() if measure_col and measure_col in df.columns else None,
            colors=colors,
            color=color,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        views, spec = task.execute()

        validate_all_views(views)

        fig, ax = plt.subplots(figsize=figsize)
        minimal_axes(ax)

        for view in views:
            draw_waterfall(ax, view)

        apply_axes_style(ax, spec)

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_waterfall workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_waterfall workflow: {e}")
        raise


def plot_waffle(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    colors: list[str] | None = None,
    rows: int | None = None,
    columns: int | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a waffle chart.

    Args:
        df: DataFrame with category and value columns.
        category_col: Name of the category column.
        value_col: Name of the value column.
        colors: Optional list of colors for each category.
        rows: Optional number of rows in the grid.
        columns: Optional number of columns in the grid.
        title: Optional plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_waffle workflow")
    try:
        task = WafflePlotTask(
            categories=df[category_col].tolist(),
            values=df[value_col].values,
            colors=colors,
            rows=rows,
            columns=columns,
            title=title,
        )
        views, spec = task.execute()

        validate_all_views(views)

        fig, ax = plt.subplots(figsize=figsize)
        minimal_axes(ax)

        for view in views:
            draw_waffle(ax, view)

        apply_axes_style(ax, spec)

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_waffle workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_waffle workflow: {e}")
        raise


def plot_dumbbell(
    df: pd.DataFrame,
    categories_col: str,
    values1_col: str,
    values2_col: str,
    label1: str | None = None,
    label2: str | None = None,
    colors: list[str] | None = None,
    color: str | None = None,
    orientation: str = "h",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a dumbbell chart.

    Args:
        df: DataFrame with categories and two value columns.
        categories_col: Name of the categories column.
        values1_col: Name of the first values column.
        values2_col: Name of the second values column.
        label1: Optional label for first set.
        label2: Optional label for second set.
        colors: Optional list of colors for each category.
        color: Optional single color for all dumbbells.
        orientation: 'h' for horizontal (default) or 'v' for vertical.
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_dumbbell workflow")
    try:
        task = DumbbellPlotTask(
            categories=df[categories_col].tolist(),
            values1=df[values1_col].values,
            values2=df[values2_col].values,
            label1=label1,
            label2=label2,
            colors=colors,
            color=color,
            orientation=orientation,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        views, spec = task.execute()

        validate_all_views(views)

        fig, ax = plt.subplots(figsize=figsize)
        minimal_axes(ax)

        for view in views:
            draw_dumbbell(ax, view)

        apply_axes_style(ax, spec)

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_dumbbell workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_dumbbell workflow: {e}")
        raise


def plot_range(
    df: pd.DataFrame,
    categories_col: str,
    values1_col: str,
    values2_col: str,
    label1: str | None = None,
    label2: str | None = None,
    color: str | None = None,
    orientation: str = "h",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a range chart (similar to dumbbell but different styling).

    Args:
        df: DataFrame with categories and two value columns.
        categories_col: Name of the categories column.
        values1_col: Name of the first values column.
        values2_col: Name of the second values column.
        label1: Optional label for first set.
        label2: Optional label for second set.
        color: Optional color for the range.
        orientation: 'h' for horizontal (default) or 'v' for vertical.
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_range workflow")
    try:
        task = RangePlotTask(
            categories=df[categories_col].tolist(),
            values1=df[values1_col].values,
            values2=df[values2_col].values,
            label1=label1,
            label2=label2,
            color=color,
            orientation=orientation,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        views, spec = task.execute()

        validate_all_views(views)

        fig, ax = plt.subplots(figsize=figsize)
        minimal_axes(ax)

        for view in views:
            draw_range(ax, view)

        apply_axes_style(ax, spec)

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_range workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_range workflow: {e}")
        raise


def plot_lollipop(
    df: pd.DataFrame,
    categories_col: str,
    values_col: str,
    color: str | None = None,
    marker: str | None = None,
    linewidth: float | None = None,
    horizontal: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a lollipop chart.

    Args:
        df: DataFrame with categories and values columns.
        categories_col: Name of the categories column.
        values_col: Name of the values column.
        color: Optional color for the lollipops.
        marker: Optional marker style for the lollipop heads.
        linewidth: Optional line width.
        horizontal: If True, create horizontal lollipops.
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_lollipop workflow")
    try:
        task = LollipopPlotTask(
            categories=df[categories_col].tolist(),
            values=df[values_col].values,
            color=color,
            marker=marker,
            linewidth=linewidth,
            horizontal=horizontal,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        views, spec = task.execute()

        validate_all_views(views)

        fig, ax = plt.subplots(figsize=figsize)
        minimal_axes(ax)

        for view in views:
            draw_lollipop(ax, view)

        apply_axes_style(ax, spec)

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_lollipop workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_lollipop workflow: {e}")
        raise


def plot_slope(
    data_frame: pd.DataFrame | None = None,
    x: str | None = None,
    y: str | None = None,
    group: str | None = None,
    df: pd.DataFrame | None = None,
    categories_col: str | None = None,
    values_cols: list[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colors: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a slope chart.

    Can be called in two ways:
    1. With long format: data_frame with x, y, and group columns
    2. With wide format: df with categories_col and values_cols (list of column names)

    Args:
        data_frame: DataFrame with x, y, and group columns (long format).
        x: Name of the x column (typically time periods) - for long format.
        y: Name of the y column (values) - for long format.
        group: Name of the group column (categories to track) - for long format.
        df: DataFrame with categories column and value columns (wide format).
        categories_col: Name of the categories column (for wide format).
        values_cols: List of column names for values (for wide format).
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        colors: Optional dictionary mapping group names to colors.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_slope workflow")
    try:
        # Determine format and convert to groups_dict
        if df is not None and categories_col is not None and values_cols is not None:
            # Wide format: DataFrame with categories column and value columns
            if categories_col not in df.columns:
                raise ValueError(f"Column '{categories_col}' not found in DataFrame")
            if len(df) != 2:
                raise ValueError("DataFrame must have exactly 2 rows for slope chart")

            groups_dict = {}
            for col in values_cols:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                groups_dict[col] = df[col].values

            # Use categories column as x values
            x_unique = df[categories_col].values
            if len(x_unique) != 2:
                raise ValueError("categories_col must have exactly 2 unique values for slope chart")
            x_unique = np.array(x_unique)
        elif data_frame is not None and x is not None and y is not None and group is not None:
            # Long format: DataFrame with x, y, and group columns
            groups_dict = {}
            for group_name, group_df in data_frame.groupby(group):
                x_vals = group_df[x].values
                y_vals = group_df[y].values
                if len(x_vals) != 2 or len(y_vals) != 2:
                    raise ValueError(f"Group '{group_name}' must have exactly 2 data points")
                groups_dict[group_name] = y_vals

            # Get unique x values (should be 2)
            x_unique = sorted(data_frame[x].unique())
            if len(x_unique) != 2:
                raise ValueError("x column must have exactly 2 unique values for slope chart")
            x_unique = np.array(x_unique)
        else:
            raise ValueError("Must provide either (data_frame, x, y, group) or (df, categories_col, values_cols)")

        task = SlopePlotTask(
            x=x_unique,
            groups=groups_dict,
            colors=colors,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        views, spec = task.execute()

        validate_all_views(views)

        fig, ax = plt.subplots(figsize=figsize)
        minimal_axes(ax)

        for view in views:
            draw_slope(ax, view)

        apply_axes_style(ax, spec)

        # Add legend
        ax.legend()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_slope workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_slope workflow: {e}")
        raise


def plot_metric(
    title: str,
    value: float,
    delta: float | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    value_color: str | None = None,
    delta_color: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (6, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a metric display.

    Args:
        title: Title text.
        value: Main value to display.
        delta: Optional change value (positive or negative).
        prefix: Optional prefix for value (e.g., '$').
        suffix: Optional suffix for value (e.g., '%').
        value_color: Optional color for the value.
        delta_color: Optional color for the delta (defaults based on sign).
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_metric workflow")
    try:
        task = MetricPlotTask(
            title=title,
            value=value,
            delta=delta,
            prefix=prefix,
            suffix=suffix,
            value_color=value_color,
            delta_color=delta_color,
        )
        views, spec = task.execute()

        validate_all_views(views)

        fig, ax = plt.subplots(figsize=figsize)
        minimal_axes(ax)

        for view in views:
            draw_metric(ax, view)

        apply_axes_style(ax, spec)

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_metric workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_metric workflow: {e}")
        raise


def plot_box(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colors: list[str] | None = None,
    color: str | None = None,
    show_outliers: bool = True,
    show_means: bool = False,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a box plot.

    Args:
        data: DataFrame with x (category) and y (value) columns.
        x: Name of the category column.
        y: Name of the value column.
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        colors: Optional list of colors for each box.
        color: Optional single color for all boxes.
        show_outliers: If True, show outliers (default True).
        show_means: If True, show means (default False).
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_box workflow")
    try:
        task = BoxPlotTask(
            data=data,
            x=x,
            y=y,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            colors=colors,
            color=color,
            show_outliers=show_outliers,
            show_means=show_means,
        )
        views, spec = task.execute()

        validate_all_views(views)

        fig, ax = plt.subplots(figsize=figsize)
        minimal_axes(ax)

        for view in views:
            draw_box(ax, view)

        apply_axes_style(ax, spec)

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_box workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_box workflow: {e}")
        raise


def plot_violin(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colors: list[str] | None = None,
    color: str | None = None,
    show_means: bool = False,
    show_medians: bool = True,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a violin plot.

    Args:
        data: DataFrame with x (category) and y (value) columns.
        x: Name of the category column.
        y: Name of the value column.
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        colors: Optional list of colors for each violin.
        color: Optional single color for all violins.
        show_means: If True, show means (default False).
        show_medians: If True, show medians (default True).
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_violin workflow")
    try:
        task = ViolinPlotTask(
            data=data,
            x=x,
            y=y,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            colors=colors,
            color=color,
            show_means=show_means,
            show_medians=show_medians,
        )
        views, spec = task.execute()

        validate_all_views(views)

        fig, ax = plt.subplots(figsize=figsize)
        minimal_axes(ax)

        for view in views:
            draw_violin(ax, view)

        apply_axes_style(ax, spec)

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_violin workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_violin workflow: {e}")
        raise


def plot_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    size: str | None = None,
    label: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    alpha: float | None = None,
    marker: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a scatter plot with optional color and size mappings.

    Args:
        data: DataFrame with x and y columns, and optionally color/size columns.
        x: Name of the x column.
        y: Name of the y column.
        color: Optional name of column to use for color mapping.
        size: Optional name of column to use for size mapping.
        label: Optional label for the scatter (for legend).
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        alpha: Optional transparency.
        marker: Optional marker style.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_scatter workflow")
    try:
        task = ScatterPlotTask(
            data=data,
            x=x,
            y=y,
            color=color,
            size=size,
            label=label,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            alpha=alpha,
            marker=marker,
        )
        views, spec = task.execute()

        validate_all_views(views)

        fig, ax = plt.subplots(figsize=figsize)
        minimal_axes(ax)

        for view in views:
            draw_scatter(ax, view)

        apply_axes_style(ax, spec)

        # Add legend if label provided
        if label:
            ax.legend()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_scatter workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_scatter workflow: {e}")
        raise


def plot_correlation(
    data: pd.DataFrame,
    method: str = "pearson",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str | None = None,
    annotate: bool = True,
    fmt: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a correlation heatmap.

    Args:
        data: DataFrame to compute correlation matrix from.
        method: Correlation method ('pearson', 'kendall', 'spearman').
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        cmap: Optional colormap name (defaults to 'RdYlGn').
        annotate: If True, annotate cells with correlation values.
        fmt: Optional format string for annotations.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_correlation workflow")
    try:
        task = CorrelationPlotTask(
            data=data,
            method=method,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            cmap=cmap,
            annotate=annotate,
            fmt=fmt,
        )
        views, spec = task.execute()

        validate_all_views(views)

        fig, ax = plt.subplots(figsize=figsize)
        minimal_axes(ax)

        for view in views:
            draw_heatmap(ax, view)

        apply_axes_style(ax, spec)

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_correlation workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_correlation workflow: {e}")
        raise


def plot_forecast_comparison(
    actual: pd.Series | np.ndarray,
    forecasts: dict[str, pd.Series | np.ndarray],
    intervals: dict[str, tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot forecast comparison with actual data and multiple forecasts.

    Args:
        actual: Actual time series data.
        forecasts: Dictionary mapping forecast names to forecast arrays/Series.
        intervals: Optional dictionary mapping forecast names to (lower, upper) interval tuples.
        title: Optional plot title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple (width, height).

    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    logger.info("Starting plot_forecast_comparison workflow")
    try:
        task = ForecastComparisonPlotTask(
            actual=actual,
            forecasts=forecasts,
            intervals=intervals,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        views, spec = task.execute()

        validate_all_views(views)

        fig, ax = plt.subplots(figsize=figsize)
        minimal_axes(ax)

        # Draw bands first (so they're behind lines)
        bands = [v for v in views if hasattr(v, "y_lower")]
        series = [v for v in views if not hasattr(v, "y_lower")]

        for view in bands:
            draw_band(ax, view)

        for view in series:
            if hasattr(view, "s"):  # ScatterView
                draw_scatter(ax, view)
            else:  # SeriesView
                draw_series(ax, view)

        apply_axes_style(ax, spec)

        # Add legend
        ax.legend()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        logger.info("Completed plot_forecast_comparison workflow")
        return fig, ax

    except Exception as e:
        logger.error(f"Error in plot_forecast_comparison workflow: {e}")
        raise

