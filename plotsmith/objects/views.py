"""Immutable dataclasses representing plot-ready data and plot specs."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SeriesView:
    """Immutable view of a time series for plotting.

    Attributes:
        x: X-axis values (e.g., time indices or timestamps).
        y: Y-axis values (the series data).
        label: Optional label for the series (for legend).
        marker: Optional marker style (e.g., 'o', 's', '-').
        linewidth: Optional line width.
        alpha: Optional transparency (0.0 to 1.0).
    """

    x: np.ndarray
    y: np.ndarray
    label: Optional[str] = None
    marker: Optional[str] = None
    linewidth: Optional[float] = None
    alpha: Optional[float] = None


@dataclass(frozen=True)
class BandView:
    """Immutable view of a confidence band or shaded region.

    Attributes:
        x: X-axis values (must align with y_lower and y_upper).
        y_lower: Lower bound of the band.
        y_upper: Upper bound of the band.
        label: Optional label for the band (for legend).
        alpha: Optional transparency (0.0 to 1.0).
    """

    x: np.ndarray
    y_lower: np.ndarray
    y_upper: np.ndarray
    label: Optional[str] = None
    alpha: Optional[float] = None


@dataclass(frozen=True)
class ScatterView:
    """Immutable view of scatter plot data.

    Attributes:
        x: X-axis values.
        y: Y-axis values.
        label: Optional label for the scatter (for legend).
        marker: Optional marker style (e.g., 'o', 's').
        alpha: Optional transparency (0.0 to 1.0).
        s: Optional marker size.
    """

    x: np.ndarray
    y: np.ndarray
    label: Optional[str] = None
    marker: Optional[str] = None
    alpha: Optional[float] = None
    s: Optional[float] = None


@dataclass(frozen=True)
class HistogramView:
    """Immutable view of histogram data.

    Attributes:
        values: Array of values to histogram.
        bins: Optional number of bins or bin edges.
        label: Optional label for the histogram (for legend).
        color: Optional fill color.
        edgecolor: Optional edge color.
        alpha: Optional transparency (0.0 to 1.0).
    """

    values: np.ndarray
    bins: Optional[int | np.ndarray] = None
    label: Optional[str] = None
    color: Optional[str] = None
    edgecolor: Optional[str] = None
    alpha: Optional[float] = None


@dataclass(frozen=True)
class BarView:
    """Immutable view of bar chart data.

    Attributes:
        x: X-axis positions (categories or positions).
        height: Bar heights.
        label: Optional label for the bars (for legend).
        color: Optional fill color.
        edgecolor: Optional edge color.
        alpha: Optional transparency (0.0 to 1.0).
        horizontal: If True, create horizontal bars.
    """

    x: np.ndarray | list[str]
    height: np.ndarray
    label: Optional[str] = None
    color: Optional[str] = None
    edgecolor: Optional[str] = None
    alpha: Optional[float] = None
    horizontal: bool = False


@dataclass(frozen=True)
class HeatmapView:
    """Immutable view of heatmap data.

    Attributes:
        data: 2D array of values to display as heatmap.
        x_labels: Optional labels for x-axis (columns).
        y_labels: Optional labels for y-axis (rows).
        cmap: Optional colormap name.
        vmin: Optional minimum value for color scale.
        vmax: Optional maximum value for color scale.
        annotate: If True, annotate cells with values.
        fmt: Optional format string for annotations.
    """

    data: np.ndarray
    x_labels: Optional[list[str] | np.ndarray] = None
    y_labels: Optional[list[str] | np.ndarray] = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    annotate: bool = False
    fmt: Optional[str] = None


@dataclass(frozen=True)
class FigureSpec:
    """Immutable specification for figure and axes styling.

    Attributes:
        title: Optional figure/axes title.
        xlabel: Optional X-axis label.
        ylabel: Optional Y-axis label.
        xlim: Optional tuple of (xmin, xmax).
        ylim: Optional tuple of (ymin, ymax).
    """

    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    xlim: Optional[tuple[float, float]] = None
    ylim: Optional[tuple[float, float]] = None

