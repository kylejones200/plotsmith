"""Tasks that convert user data into Layer 1 view objects."""

import numpy as np
import pandas as pd

from plotsmith.objects.views import (
    BandView,
    BarView,
    BoxView,
    DumbbellView,
    FigureSpec,
    HeatmapView,
    HistogramView,
    LollipopView,
    MetricView,
    RangeView,
    ScatterView,
    SeriesView,
    SlopeView,
    ViolinView,
    WaffleView,
    WaterfallView,
)
from plotsmith.utils.validation import (
    validate_dataframe_columns,
)


class TimeseriesPlotTask:
    """Task to create views for time series plotting.

    Accepts pandas Series or DataFrame and produces SeriesView objects
    plus a FigureSpec.
    """

    def __init__(
        self,
        data: pd.Series | pd.DataFrame,
        bands: dict[str, tuple[pd.Series, pd.Series]] | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
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
            raise TypeError(
                f"data must be pandas Series or DataFrame, got {type(self.data)}"
            )

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
            views.append(
                BandView(x=x_band, y_lower=y_lower, y_upper=y_upper, label=band_name)
            )

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
        fold_id_col: str | None = "fold_id",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
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
            available = ", ".join(self.results.columns.tolist())
            raise ValueError(
                f"Column '{self.y_true_col}' not found in results. "
                f"Available columns: {available}"
            )
        if self.y_pred_col not in self.results.columns:
            available = ", ".join(self.results.columns.tolist())
            raise ValueError(
                f"Column '{self.y_pred_col}' not found in results. "
                f"Available columns: {available}"
            )

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
        bins: int | np.ndarray | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        label: str | None = None,
        color: str | None = None,
        edgecolor: str | None = None,
        alpha: float | None = None,
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
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        label: str | None = None,
        color: str | None = None,
        edgecolor: str | None = None,
        alpha: float | None = None,
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

        self.height = (
            np.asarray(height) if not isinstance(height, np.ndarray) else height
        )
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
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        annotate: bool = False,
        fmt: str | None = None,
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
        x: np.ndarray | pd.Series | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
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
        self.actual = (
            np.asarray(actual) if not isinstance(actual, np.ndarray) else actual
        )
        self.predicted = (
            np.asarray(predicted)
            if not isinstance(predicted, np.ndarray)
            else predicted
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
            views.append(
                ScatterView(x=self.actual, y=self.predicted, label="Residuals")
            )
        elif self.plot_type == "series":
            # Time series plot: residuals over time
            views.append(SeriesView(x=self.x, y=self.residuals, label="Residuals"))
        else:
            raise ValueError(
                f"plot_type must be 'scatter' or 'series', got {self.plot_type}"
            )

        spec = FigureSpec(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        return views, spec


class WaterfallPlotTask:
    """Task to create views for waterfall chart plotting.

    Accepts categories and values and produces WaterfallView objects plus a FigureSpec.
    """

    def __init__(
        self,
        categories: list[str] | np.ndarray,
        values: np.ndarray | list[float],
        measures: list[str] | np.ndarray | None = None,
        colors: list[str] | np.ndarray | None = None,
        color: str | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ):
        """Initialize the waterfall plot task.

        Args:
            categories: Category labels for each bar.
            values: Values for each category.
            measures: Optional list of measure types ('absolute', 'relative', 'total').
            colors: Optional list of colors for each bar.
            color: Optional single color for all bars.
            title: Optional plot title.
            xlabel: Optional X-axis label.
            ylabel: Optional Y-axis label.
        """
        self.categories = (
            list(categories) if isinstance(categories, np.ndarray) else categories
        )
        self.values = (
            np.asarray(values) if not isinstance(values, np.ndarray) else values
        )
        self.measures = (
            list(measures)
            if measures is not None and isinstance(measures, np.ndarray)
            else measures
        )
        self.colors = (
            list(colors)
            if colors is not None and isinstance(colors, np.ndarray)
            else colors
        )
        self.color = color
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def execute(self) -> tuple[list[WaterfallView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of WaterfallView objects, FigureSpec).
        """
        view = WaterfallView(
            categories=self.categories,
            values=self.values,
            measures=self.measures,
            colors=self.colors,
            color=self.color,
        )

        spec = FigureSpec(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        return [view], spec


class WafflePlotTask:
    """Task to create views for waffle chart plotting.

    Accepts categories and values and produces WaffleView objects plus a FigureSpec.
    """

    def __init__(
        self,
        categories: list[str] | np.ndarray,
        values: np.ndarray | list[float],
        colors: list[str] | np.ndarray | None = None,
        rows: int | None = None,
        columns: int | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ):
        """Initialize the waffle plot task.

        Args:
            categories: Category labels.
            values: Values for each category.
            colors: Optional list of colors for each category.
            rows: Optional number of rows in the grid.
            columns: Optional number of columns in the grid.
            title: Optional plot title.
            xlabel: Optional X-axis label.
            ylabel: Optional Y-axis label.
        """
        self.categories = (
            list(categories) if isinstance(categories, np.ndarray) else categories
        )
        self.values = (
            np.asarray(values) if not isinstance(values, np.ndarray) else values
        )
        self.colors = (
            list(colors)
            if colors is not None and isinstance(colors, np.ndarray)
            else colors
        )
        self.rows = rows
        self.columns = columns
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def execute(self) -> tuple[list[WaffleView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of WaffleView objects, FigureSpec).
        """
        view = WaffleView(
            categories=self.categories,
            values=self.values,
            colors=self.colors,
            rows=self.rows,
            columns=self.columns,
        )

        spec = FigureSpec(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        return [view], spec


class DumbbellPlotTask:
    """Task to create views for dumbbell chart plotting.

    Accepts categories and two sets of values and produces DumbbellView objects plus a FigureSpec.
    """

    def __init__(
        self,
        categories: list[str] | np.ndarray,
        values1: np.ndarray | list[float],
        values2: np.ndarray | list[float],
        label1: str | None = None,
        label2: str | None = None,
        colors: list[str] | np.ndarray | None = None,
        color: str | None = None,
        orientation: str = "h",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ):
        """Initialize the dumbbell plot task.

        Args:
            categories: Category labels.
            values1: First set of values (left/start points).
            values2: Second set of values (right/end points).
            label1: Optional label for first set.
            label2: Optional label for second set.
            colors: Optional list of colors for each category.
            color: Optional single color for all dumbbells.
            orientation: 'h' for horizontal (default) or 'v' for vertical.
            title: Optional plot title.
            xlabel: Optional X-axis label.
            ylabel: Optional Y-axis label.
        """
        self.categories = (
            list(categories) if isinstance(categories, np.ndarray) else categories
        )
        self.values1 = (
            np.asarray(values1) if not isinstance(values1, np.ndarray) else values1
        )
        self.values2 = (
            np.asarray(values2) if not isinstance(values2, np.ndarray) else values2
        )
        self.label1 = label1
        self.label2 = label2
        self.colors = (
            list(colors)
            if colors is not None and isinstance(colors, np.ndarray)
            else colors
        )
        self.color = color
        self.orientation = orientation
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def execute(self) -> tuple[list[DumbbellView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of DumbbellView objects, FigureSpec).
        """
        if len(self.values1) != len(self.values2):
            raise ValueError("values1 and values2 must have the same length")
        if len(self.categories) != len(self.values1):
            raise ValueError("categories must have the same length as values")

        view = DumbbellView(
            categories=self.categories,
            values1=self.values1,
            values2=self.values2,
            label1=self.label1,
            label2=self.label2,
            colors=self.colors,
            color=self.color,
            orientation=self.orientation,
        )

        spec = FigureSpec(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        return [view], spec


class RangePlotTask:
    """Task to create views for range chart plotting.

    Accepts categories and two sets of values and produces RangeView objects plus a FigureSpec.
    """

    def __init__(
        self,
        categories: list[str] | np.ndarray,
        values1: np.ndarray | list[float],
        values2: np.ndarray | list[float],
        label1: str | None = None,
        label2: str | None = None,
        color: str | None = None,
        orientation: str = "h",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ):
        """Initialize the range plot task.

        Args:
            categories: Category labels.
            values1: First set of values (left/start points).
            values2: Second set of values (right/end points).
            label1: Optional label for first set.
            label2: Optional label for second set.
            color: Optional color for the range.
            orientation: 'h' for horizontal (default) or 'v' for vertical.
            title: Optional plot title.
            xlabel: Optional X-axis label.
            ylabel: Optional Y-axis label.
        """
        self.categories = (
            list(categories) if isinstance(categories, np.ndarray) else categories
        )
        self.values1 = (
            np.asarray(values1) if not isinstance(values1, np.ndarray) else values1
        )
        self.values2 = (
            np.asarray(values2) if not isinstance(values2, np.ndarray) else values2
        )
        self.label1 = label1
        self.label2 = label2
        self.color = color
        self.orientation = orientation
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def execute(self) -> tuple[list[RangeView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of RangeView objects, FigureSpec).
        """
        if len(self.values1) != len(self.values2):
            raise ValueError("values1 and values2 must have the same length")
        if len(self.categories) != len(self.values1):
            raise ValueError("categories must have the same length as values")

        view = RangeView(
            categories=self.categories,
            values1=self.values1,
            values2=self.values2,
            label1=self.label1,
            label2=self.label2,
            color=self.color,
            orientation=self.orientation,
        )

        spec = FigureSpec(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        return [view], spec


class LollipopPlotTask:
    """Task to create views for lollipop chart plotting.

    Accepts categories and values and produces LollipopView objects plus a FigureSpec.
    """

    def __init__(
        self,
        categories: list[str] | np.ndarray,
        values: np.ndarray | list[float],
        color: str | None = None,
        marker: str | None = None,
        linewidth: float | None = None,
        horizontal: bool = False,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ):
        """Initialize the lollipop plot task.

        Args:
            categories: Category labels.
            values: Values for each category.
            color: Optional color for the lollipops.
            marker: Optional marker style for the lollipop heads.
            linewidth: Optional line width.
            horizontal: If True, create horizontal lollipops.
            title: Optional plot title.
            xlabel: Optional X-axis label.
            ylabel: Optional Y-axis label.
        """
        self.categories = (
            list(categories) if isinstance(categories, np.ndarray) else categories
        )
        self.values = (
            np.asarray(values) if not isinstance(values, np.ndarray) else values
        )
        self.color = color
        self.marker = marker
        self.linewidth = linewidth
        self.horizontal = horizontal
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def execute(self) -> tuple[list[LollipopView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of LollipopView objects, FigureSpec).
        """
        if len(self.categories) != len(self.values):
            raise ValueError("categories must have the same length as values")

        view = LollipopView(
            categories=self.categories,
            values=self.values,
            color=self.color,
            marker=self.marker,
            linewidth=self.linewidth,
            horizontal=self.horizontal,
        )

        spec = FigureSpec(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        return [view], spec


class SlopePlotTask:
    """Task to create views for slope chart plotting.

    Accepts x values and groups of y values and produces SlopeView objects plus a FigureSpec.
    """

    def __init__(
        self,
        x: np.ndarray | list[float],
        groups: dict[str, np.ndarray | list[float]],
        colors: dict[str, str] | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ):
        """Initialize the slope plot task.

        Args:
            x: X-axis values (typically two values for start and end).
            groups: Dictionary mapping group names to y-values arrays.
            colors: Optional dictionary mapping group names to colors.
            title: Optional plot title.
            xlabel: Optional X-axis label.
            ylabel: Optional Y-axis label.
        """
        self.x = np.asarray(x) if not isinstance(x, np.ndarray) else x
        self.groups = {
            k: (np.asarray(v) if not isinstance(v, np.ndarray) else v)
            for k, v in groups.items()
        }
        self.colors = colors
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def execute(self) -> tuple[list[SlopeView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of SlopeView objects, FigureSpec).
        """
        if len(self.x) != 2:
            raise ValueError("x must have exactly 2 values for slope chart")
        for group_name, y_values in self.groups.items():
            if len(y_values) != 2:
                raise ValueError(f"Group '{group_name}' must have exactly 2 y values")

        view = SlopeView(x=self.x, groups=self.groups, colors=self.colors)

        spec = FigureSpec(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        return [view], spec


class MetricPlotTask:
    """Task to create views for metric display plotting.

    Accepts title, value, and optional delta and produces MetricView objects plus a FigureSpec.
    """

    def __init__(
        self,
        title: str,
        value: float,
        delta: float | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        value_color: str | None = None,
        delta_color: str | None = None,
    ):
        """Initialize the metric plot task.

        Args:
            title: Title text.
            value: Main value to display.
            delta: Optional change value (positive or negative).
            prefix: Optional prefix for value (e.g., '$').
            suffix: Optional suffix for value (e.g., '%').
            value_color: Optional color for the value.
            delta_color: Optional color for the delta (defaults based on sign).
        """
        self.title = title
        self.value = value
        self.delta = delta
        self.prefix = prefix
        self.suffix = suffix
        self.value_color = value_color
        self.delta_color = delta_color

    def execute(self) -> tuple[list[MetricView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of MetricView objects, FigureSpec).
        """
        view = MetricView(
            title=self.title,
            value=self.value,
            delta=self.delta,
            prefix=self.prefix,
            suffix=self.suffix,
            value_color=self.value_color,
            delta_color=self.delta_color,
        )

        spec = FigureSpec()

        return [view], spec


class BoxPlotTask:
    """Task to create views for box plot plotting.

    Accepts data grouped by category and produces BoxView objects plus a FigureSpec.
    """

    def __init__(
        self,
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
    ):
        """Initialize the box plot task.

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
        """
        self.data = data
        self.x = x
        self.y = y
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.colors = colors
        self.color = color
        self.show_outliers = show_outliers
        self.show_means = show_means

    def execute(self) -> tuple[list[BoxView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of BoxView objects, FigureSpec).
        """
        validate_dataframe_columns(self.data, [self.x, self.y], df_name="data")

        # Group data by category - vectorized with groupby
        grouped = self.data.groupby(self.x)[self.y]
        data_list = [group.values for _, group in grouped]
        labels = [str(cat) for cat in sorted(self.data[self.x].unique())]

        view = BoxView(
            data=data_list,
            labels=labels,
            colors=self.colors,
            color=self.color,
            show_outliers=self.show_outliers,
            show_means=self.show_means,
        )

        spec = FigureSpec(
            title=self.title, xlabel=self.xlabel or self.x, ylabel=self.ylabel or self.y
        )

        return [view], spec


class ViolinPlotTask:
    """Task to create views for violin plot plotting.

    Accepts data grouped by category and produces ViolinView objects plus a FigureSpec.
    """

    def __init__(
        self,
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
    ):
        """Initialize the violin plot task.

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
        """
        self.data = data
        self.x = x
        self.y = y
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.colors = colors
        self.color = color
        self.show_means = show_means
        self.show_medians = show_medians

    def execute(self) -> tuple[list[ViolinView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of ViolinView objects, FigureSpec).
        """
        validate_dataframe_columns(self.data, [self.x, self.y], df_name="data")

        # Group data by category - vectorized with groupby
        grouped = self.data.groupby(self.x)[self.y]
        data_list = [group.values for _, group in grouped]
        labels = [str(cat) for cat in sorted(self.data[self.x].unique())]

        view = ViolinView(
            data=data_list,
            labels=labels,
            colors=self.colors,
            color=self.color,
            show_means=self.show_means,
            show_medians=self.show_medians,
        )

        spec = FigureSpec(
            title=self.title, xlabel=self.xlabel or self.x, ylabel=self.ylabel or self.y
        )

        return [view], spec


class ScatterPlotTask:
    """Task to create views for scatter plot plotting with enhanced features.

    Accepts DataFrame with x, y, and optional color/size columns.
    """

    def __init__(
        self,
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
    ):
        """Initialize the scatter plot task.

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
        """
        self.data = data
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.label = label
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.alpha = alpha
        self.marker = marker

    def execute(self) -> tuple[list[ScatterView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of ScatterView objects, FigureSpec).
        """
        required_cols = [self.x, self.y]
        if self.color:
            required_cols.append(self.color)
        if self.size:
            required_cols.append(self.size)
        validate_dataframe_columns(self.data, required_cols, df_name="data")

        x_vals = self.data[self.x].values
        y_vals = self.data[self.y].values

        # Handle color mapping - more Pythonic
        c_vals = (
            self.data[self.color].values
            if (self.color and self.color in self.data.columns)
            else None
        )

        # Handle size mapping - more Pythonic
        s_vals = (
            self.data[self.size].values
            if (self.size and self.size in self.data.columns)
            else None
        )

        view = ScatterView(
            x=x_vals,
            y=y_vals,
            label=self.label,
            marker=self.marker,
            alpha=self.alpha,
            s=s_vals,
            c=c_vals,
        )

        spec = FigureSpec(
            title=self.title, xlabel=self.xlabel or self.x, ylabel=self.ylabel or self.y
        )

        return [view], spec


class CorrelationPlotTask:
    """Task to create views for correlation heatmap plotting.

    Accepts DataFrame and produces HeatmapView with correlation matrix.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        method: str = "pearson",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        cmap: str | None = None,
        annotate: bool = True,
        fmt: str | None = None,
    ):
        """Initialize the correlation plot task.

        Args:
            data: DataFrame to compute correlation matrix from.
            method: Correlation method ('pearson', 'kendall', 'spearman').
            title: Optional plot title.
            xlabel: Optional X-axis label.
            ylabel: Optional Y-axis label.
            cmap: Optional colormap name.
            annotate: If True, annotate cells with correlation values.
            fmt: Optional format string for annotations.
        """
        self.data = data
        self.method = method
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cmap = cmap or "RdYlGn"
        self.annotate = annotate
        self.fmt = fmt or ".2f"

    def execute(self) -> tuple[list[HeatmapView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of HeatmapView objects, FigureSpec).
        """
        # Compute correlation matrix
        corr_matrix = self.data.corr(method=self.method)

        view = HeatmapView(
            data=corr_matrix.values,
            x_labels=corr_matrix.columns.tolist(),
            y_labels=corr_matrix.index.tolist(),
            cmap=self.cmap,
            vmin=-1.0,
            vmax=1.0,
            annotate=self.annotate,
            fmt=self.fmt,
        )

        title = self.title or f"Correlation Matrix ({self.method})"
        spec = FigureSpec(title=title, xlabel=self.xlabel, ylabel=self.ylabel)

        return [view], spec


class ForecastComparisonPlotTask:
    """Task to create views for forecast comparison plotting.

    Accepts actual data and dictionary of forecasts, produces SeriesView and BandView objects.
    """

    def __init__(
        self,
        actual: pd.Series | np.ndarray,
        forecasts: dict[str, pd.Series | np.ndarray],
        intervals: dict[str, tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]]
        | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ):
        """Initialize the forecast comparison plot task.

        Args:
            actual: Actual time series data.
            forecasts: Dictionary mapping forecast names to forecast arrays/Series.
            intervals: Optional dictionary mapping forecast names to (lower, upper) interval tuples.
            title: Optional plot title.
            xlabel: Optional X-axis label.
            ylabel: Optional Y-axis label.
        """
        self.actual = actual
        self.forecasts = forecasts
        self.intervals = intervals or {}
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def execute(self) -> tuple[list[SeriesView | BandView], FigureSpec]:
        """Execute the task and return views and figure spec.

        Returns:
            Tuple of (list of view objects, FigureSpec).
        """
        views: list[SeriesView | BandView] = []

        # Convert actual to SeriesView
        if isinstance(self.actual, pd.Series):
            x = self.actual.index.values
            y = self.actual.values
        else:
            x = np.arange(len(self.actual))
            y = np.asarray(self.actual)

        views.append(SeriesView(x=x, y=y, label="Actual", linewidth=2))

        # Add forecasts
        for name, forecast in self.forecasts.items():
            if isinstance(forecast, pd.Series):
                forecast_x = forecast.index.values
                forecast_y = forecast.values
            else:
                forecast_x = x  # Use same x as actual
                forecast_y = np.asarray(forecast)

            views.append(
                SeriesView(
                    x=forecast_x, y=forecast_y, label=name, linewidth=1.5, alpha=0.7
                )
            )

            # Add intervals if provided
            if name in self.intervals:
                lower, upper = self.intervals[name]
                if isinstance(lower, pd.Series):
                    lower_x = lower.index.values
                    lower_y = lower.values
                    upper_y = upper.values if isinstance(upper, pd.Series) else np.asarray(upper)
                else:
                    lower_x = forecast_x
                    lower_y = np.asarray(lower)
                    upper_y = np.asarray(upper)

                views.append(
                    BandView(
                        x=lower_x,
                        y_lower=lower_y,
                        y_upper=upper_y,
                        label=f"{name} CI",
                        alpha=0.2,
                    )
                )

        spec = FigureSpec(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        return views, spec
