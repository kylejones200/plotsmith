"""Thin wrappers matching old Signalplot function names."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plotsmith.workflows.workflows import plot_backtest, plot_timeseries


def plot_series(
    data: pd.Series | pd.DataFrame,
    bands: dict[str, tuple[pd.Series, pd.Series]] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a time series (deprecated, use plot_timeseries).

    This function is deprecated. Use plotsmith.plot_timeseries instead.

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
    warnings.warn(
        "plot_series is deprecated. Use plotsmith.plot_timeseries instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_timeseries(
        data=data,
        bands=bands,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=save_path,
        figsize=figsize,
    )


def plot_prediction_scatter(
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
    """Plot prediction scatter plot (deprecated, use plot_backtest).

    This function is deprecated. Use plotsmith.plot_backtest instead.

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
    warnings.warn(
        "plot_prediction_scatter is deprecated. Use plotsmith.plot_backtest instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_backtest(
        results=results,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        fold_id_col=fold_id_col,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=save_path,
        figsize=figsize,
    )
