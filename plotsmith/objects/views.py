"""Immutable dataclasses representing plot-ready data and plot specs."""

from dataclasses import dataclass

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
    label: str | None = None
    marker: str | None = None
    linewidth: float | None = None
    alpha: float | None = None


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
    label: str | None = None
    alpha: float | None = None


@dataclass(frozen=True)
class ScatterView:
    """Immutable view of scatter plot data.

    Attributes:
        x: X-axis values.
        y: Y-axis values.
        label: Optional label for the scatter (for legend).
        marker: Optional marker style (e.g., 'o', 's').
        alpha: Optional transparency (0.0 to 1.0).
        s: Optional marker size (single value or array).
        c: Optional marker color (single value, array, or column name for mapping).
    """

    x: np.ndarray
    y: np.ndarray
    label: str | None = None
    marker: str | None = None
    alpha: float | None = None
    s: float | np.ndarray | None = None
    c: str | np.ndarray | None = None


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
    bins: int | np.ndarray | None = None
    label: str | None = None
    color: str | None = None
    edgecolor: str | None = None
    alpha: float | None = None


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
    label: str | None = None
    color: str | None = None
    edgecolor: str | None = None
    alpha: float | None = None
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
    x_labels: list[str] | np.ndarray | None = None
    y_labels: list[str] | np.ndarray | None = None
    cmap: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    annotate: bool = False
    fmt: str | None = None


@dataclass(frozen=True)
class WaterfallView:
    """Immutable view of waterfall chart data.

    Attributes:
        categories: Category labels for each bar.
        values: Values for each category.
        measures: Optional list of measure types ('absolute', 'relative', 'total').
        colors: Optional list of colors for each bar.
        color: Optional single color for all bars.
    """

    categories: list[str] | np.ndarray
    values: np.ndarray
    measures: list[str] | np.ndarray | None = None
    colors: list[str] | np.ndarray | None = None
    color: str | None = None


@dataclass(frozen=True)
class WaffleView:
    """Immutable view of waffle chart data.

    Attributes:
        categories: Category labels.
        values: Values for each category.
        colors: Optional list of colors for each category.
        rows: Optional number of rows in the grid.
        columns: Optional number of columns in the grid.
    """

    categories: list[str] | np.ndarray
    values: np.ndarray
    colors: list[str] | np.ndarray | None = None
    rows: int | None = None
    columns: int | None = None


@dataclass(frozen=True)
class DumbbellView:
    """Immutable view of dumbbell chart data.

    Attributes:
        categories: Category labels.
        values1: First set of values (left/start points).
        values2: Second set of values (right/end points).
        label1: Optional label for first set.
        label2: Optional label for second set.
        colors: Optional list of colors for each category.
        color: Optional single color for all dumbbells.
        orientation: 'h' for horizontal (default) or 'v' for vertical.
    """

    categories: list[str] | np.ndarray
    values1: np.ndarray
    values2: np.ndarray
    label1: str | None = None
    label2: str | None = None
    colors: list[str] | np.ndarray | None = None
    color: str | None = None
    orientation: str = "h"


@dataclass(frozen=True)
class RangeView:
    """Immutable view of range chart data (similar to dumbbell but different styling).

    Attributes:
        categories: Category labels.
        values1: First set of values (left/start points).
        values2: Second set of values (right/end points).
        label1: Optional label for first set.
        label2: Optional label for second set.
        color: Optional color for the range.
        orientation: 'h' for horizontal (default) or 'v' for vertical.
    """

    categories: list[str] | np.ndarray
    values1: np.ndarray
    values2: np.ndarray
    label1: str | None = None
    label2: str | None = None
    color: str | None = None
    orientation: str = "h"


@dataclass(frozen=True)
class LollipopView:
    """Immutable view of lollipop chart data.

    Attributes:
        categories: Category labels.
        values: Values for each category.
        color: Optional color for the lollipops.
        marker: Optional marker style for the lollipop heads.
        linewidth: Optional line width.
        horizontal: If True, create horizontal lollipops.
    """

    categories: list[str] | np.ndarray
    values: np.ndarray
    color: str | None = None
    marker: str | None = None
    linewidth: float | None = None
    horizontal: bool = False


@dataclass(frozen=True)
class SlopeView:
    """Immutable view of slope chart data.

    Attributes:
        x: X-axis values (typically two values for start and end).
        groups: Dictionary mapping group names to y-values arrays.
        colors: Optional dictionary mapping group names to colors.
    """

    x: np.ndarray
    groups: dict[str, np.ndarray]
    colors: dict[str, str] | None = None


@dataclass(frozen=True)
class MetricView:
    """Immutable view of metric display data.

    Attributes:
        title: Title text.
        value: Main value to display.
        delta: Optional change value (positive or negative).
        prefix: Optional prefix for value (e.g., '$').
        suffix: Optional suffix for value (e.g., '%').
        value_color: Optional color for the value.
        delta_color: Optional color for the delta (defaults based on sign).
    """

    title: str
    value: float
    delta: float | None = None
    prefix: str | None = None
    suffix: str | None = None
    value_color: str | None = None
    delta_color: str | None = None


@dataclass(frozen=True)
class BoxView:
    """Immutable view of box plot data.

    Attributes:
        data: List of arrays, one per box (category).
        labels: Optional labels for each box.
        positions: Optional positions for boxes on x-axis.
        colors: Optional list of colors for each box.
        color: Optional single color for all boxes.
        show_outliers: If True, show outliers (default True).
        show_means: If True, show means (default False).
    """

    data: list[np.ndarray]
    labels: list[str] | np.ndarray | None = None
    positions: np.ndarray | None = None
    colors: list[str] | np.ndarray | None = None
    color: str | None = None
    show_outliers: bool = True
    show_means: bool = False


@dataclass(frozen=True)
class ViolinView:
    """Immutable view of violin plot data.

    Attributes:
        data: List of arrays, one per violin (category).
        labels: Optional labels for each violin.
        positions: Optional positions for violins on x-axis.
        colors: Optional list of colors for each violin.
        color: Optional single color for all violins.
        show_means: If True, show means (default False).
        show_medians: If True, show medians (default True).
    """

    data: list[np.ndarray]
    labels: list[str] | np.ndarray | None = None
    positions: np.ndarray | None = None
    colors: list[str] | np.ndarray | None = None
    color: str | None = None
    show_means: bool = False
    show_medians: bool = True


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

    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
