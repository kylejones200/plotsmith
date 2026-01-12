"""Drawing functions that accept only Layer 1 objects."""

from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

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


def minimal_axes(ax: "Axes") -> None:
    """Apply minimalist axes styling.

    Removes top and right spines, keeps left and bottom spines.
    Uses serif font.

    Args:
        ax: Matplotlib axes to style.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Set serif font
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontfamily("serif")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("serif")


def draw_series(ax: "Axes", view: SeriesView) -> None:
    """Draw a time series on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: SeriesView containing the data to plot.
    """
    kwargs: dict[str, str | float] = {}
    if view.marker is not None:
        kwargs["marker"] = view.marker
    if view.linewidth is not None:
        kwargs["linewidth"] = view.linewidth
    if view.alpha is not None:
        kwargs["alpha"] = view.alpha

    ax.plot(view.x, view.y, label=view.label, **kwargs)  # type: ignore[arg-type]


def draw_band(ax: "Axes", view: BandView) -> None:
    """Draw a confidence band or shaded region on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: BandView containing the data to plot.
    """
    kwargs: dict[str, float] = {}
    if view.alpha is not None:
        kwargs["alpha"] = view.alpha

    ax.fill_between(view.x, view.y_lower, view.y_upper, label=view.label, **kwargs)  # type: ignore[arg-type]


def draw_scatter(ax: "Axes", view: ScatterView) -> None:
    """Draw a scatter plot on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: ScatterView containing the data to plot.
    """
    kwargs: dict[str, str | float | np.ndarray] = {}
    if view.marker is not None:
        kwargs["marker"] = view.marker
    if view.alpha is not None:
        kwargs["alpha"] = view.alpha
    if view.s is not None:
        kwargs["s"] = view.s
    if view.c is not None:
        kwargs["c"] = view.c

    ax.scatter(view.x, view.y, label=view.label, **kwargs)  # type: ignore[arg-type]


def draw_histogram(ax: "Axes", view: HistogramView) -> None:
    """Draw a histogram on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: HistogramView containing the data to plot.
    """
    kwargs: dict[str, str | float | int | np.ndarray] = {}
    if view.bins is not None:
        kwargs["bins"] = view.bins
    if view.color is not None:
        kwargs["color"] = view.color
    if view.edgecolor is not None:
        kwargs["edgecolor"] = view.edgecolor
    if view.alpha is not None:
        kwargs["alpha"] = view.alpha
    if view.label is not None:
        kwargs["label"] = view.label

    ax.hist(view.values, **kwargs)  # type: ignore[arg-type]


def draw_bar(ax: "Axes", view: BarView) -> None:
    """Draw a bar chart on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: BarView containing the data to plot.
    """
    kwargs: dict[str, str | float] = {}
    if view.color is not None:
        kwargs["color"] = view.color
    if view.edgecolor is not None:
        kwargs["edgecolor"] = view.edgecolor
    if view.alpha is not None:
        kwargs["alpha"] = view.alpha
    if view.label is not None:
        kwargs["label"] = view.label

    if view.horizontal:
        ax.barh(view.x, view.height, **kwargs)  # type: ignore[arg-type]
    else:
        ax.bar(view.x, view.height, **kwargs)  # type: ignore[arg-type]


def draw_heatmap(ax: "Axes", view: HeatmapView) -> None:
    """Draw a heatmap on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: HeatmapView containing the data to plot.
    """
    kwargs: dict[str, str | float] = {}
    if view.cmap is not None:
        kwargs["cmap"] = view.cmap
    if view.vmin is not None:
        kwargs["vmin"] = view.vmin
    if view.vmax is not None:
        kwargs["vmax"] = view.vmax

    im = ax.imshow(view.data, aspect="auto", **kwargs)  # type: ignore[arg-type]

    # Set labels if provided
    if view.x_labels is not None:
        ax.set_xticks(range(len(view.x_labels)))
        ax.set_xticklabels(view.x_labels)
    if view.y_labels is not None:
        ax.set_yticks(range(len(view.y_labels)))
        ax.set_yticklabels(view.y_labels)

    # Add annotations if requested - vectorized iteration
    if view.annotate:
        fmt = view.fmt or ".2f"
        rows, cols = view.data.shape
        for i, j in np.ndindex(rows, cols):
            ax.text(
                j,
                i,
                f"{view.data[i, j]:{fmt}}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    # Add colorbar
    plt.colorbar(im, ax=ax)


def apply_axes_style(ax: "Axes", spec: FigureSpec) -> None:
    """Apply figure specification to axes.

    Args:
        ax: Matplotlib axes to style.
        spec: FigureSpec containing styling information.
    """
    if spec.title is not None:
        ax.set_title(spec.title)
    if spec.xlabel is not None:
        ax.set_xlabel(spec.xlabel)
    if spec.ylabel is not None:
        ax.set_ylabel(spec.ylabel)
    if spec.xlim is not None:
        ax.set_xlim(spec.xlim)
    if spec.ylim is not None:
        ax.set_ylim(spec.ylim)


def tidy_axes(ax: "Axes") -> None:
    """Clean up axes spines and ticks following SignalPlot style.

    Removes top and right spines, colors remaining spines gray,
    and adjusts tick parameters.

    Args:
        ax: Matplotlib axes to style.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.35")
    ax.spines["bottom"].set_color("0.35")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors="0.25", width=0.8, length=3)


def force_bar_zero(ax: "Axes") -> None:
    """Force bar chart y-axis to start at zero, preserving honest scale.

    Args:
        ax: Matplotlib axes to adjust.
    """
    lo, hi = ax.get_ylim()
    if hi <= 0:
        hi = 1.0
    ax.set_ylim(bottom=0, top=hi)


def add_range_frame(ax: "Axes", x_data=None, y_data=None) -> None:
    """Shorten axes spines so they span only the data range.

    Args:
        ax: Matplotlib axes to adjust.
        x_data: Optional iterable of x data values.
        y_data: Optional iterable of y data values.
    """
    if x_data is not None:
        x_arr = np.asarray(x_data)
        x_min, x_max = float(x_arr.min()), float(x_arr.max())
        ax.spines["bottom"].set_bounds(x_min, x_max)

    if y_data is not None:
        y_arr = np.asarray(y_data)
        y_min, y_max = float(y_arr.min()), float(y_arr.max())
        ax.spines["left"].set_bounds(y_min, y_max)


def style_line_plot(ax: "Axes", *, emphasize_last: bool = False) -> None:
    """Apply a restrained hierarchy of line styles to an axes.

    Args:
        ax: Matplotlib axes to style.
        emphasize_last: If True, emphasize the last line with thicker width.
    """
    tidy_axes(ax)

    lines = ax.get_lines()
    line_colors = ["#000000", "#404040", "#808080"]  # Black, dark gray, gray
    line_styles = ["-", "--", ":"]

    for i, line in enumerate(lines):
        color = line_colors[i % len(line_colors)]
        style = line_styles[i % len(line_styles)]

        line.set_color(color)
        line.set_linestyle(style)
        line.set_linewidth(1.2)

        if emphasize_last and i == len(lines) - 1:
            line.set_linewidth(2.0)
            line.set_color("#000000")


def style_scatter_plot(ax: "Axes") -> None:
    """Style scatter collections with neutral points.

    Args:
        ax: Matplotlib axes to style.
    """
    tidy_axes(ax)

    for collection in ax.collections:
        collection.set_edgecolors("#000000")  # type: ignore[attr-defined]
        collection.set_facecolors("#FFFFFF")  # type: ignore[attr-defined]
        collection.set_linewidths(1.0)  # type: ignore[attr-defined]


def style_bar_plot(ax: "Axes", *, horizontal: bool = False) -> None:
    """Style bar charts with alternating light fills and clean edges.

    Args:
        ax: Matplotlib axes to style.
        horizontal: If True, style for horizontal bars.
    """
    tidy_axes(ax)

    for i, patch in enumerate(ax.patches):
        patch.set_facecolor("#FFFFFF" if i % 2 == 0 else "#E8E8E8")
        patch.set_edgecolor("#000000")
        patch.set_linewidth(1.0)

    if horizontal:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)


# Accent color for emphasis
ACCENT = "#d62728"  # muted red, readable in print


def direct_label(
    ax: "Axes",
    x: float,
    y: float,
    text: str,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    use_accent: bool = False,
    ha: str = "left",
    va: str = "center",
    **kwargs,
) -> None:
    """Place a direct label near a data point.

    Typical use is to label the last point of a line instead of using a legend.

    Args:
        ax: Matplotlib axes to label.
        x: X coordinate of the point.
        y: Y coordinate of the point.
        text: Label text.
        dx: X offset for label position.
        dy: Y offset for label position.
        use_accent: If True, use accent color.
        ha: Horizontal alignment.
        va: Vertical alignment.
        **kwargs: Additional arguments passed to ax.text().
    """
    color = ACCENT if use_accent else "0.1"
    ax.text(
        x + dx,
        y + dy,
        text,
        ha=ha,
        va=va,
        color=color,
        fontsize=plt.rcParams["font.size"],
        **kwargs,
    )


def note(ax: "Axes", x: float, y: float, text: str, **kwargs) -> None:
    """Attach a short note with a simple arrow, avoiding legends.

    Args:
        ax: Matplotlib axes to annotate.
        x: X coordinate of the point.
        y: Y coordinate of the point.
        text: Note text.
        **kwargs: Additional arguments passed to ax.annotate().
    """
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(10, 10),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "0.35", "linewidth": 0.8},
        color="0.15",
        **kwargs,
    )


def emphasize_last(
    ax: "Axes", x: float, y: float, *, size: float = 30.0, **kwargs
) -> None:
    """Emphasize a final point in a series using the accent color.

    Args:
        ax: Matplotlib axes to modify.
        x: X coordinate of the point.
        y: Y coordinate of the point.
        size: Marker size.
        **kwargs: Additional arguments passed to ax.scatter().
    """
    ax.scatter([x], [y], s=size, color=ACCENT, zorder=3, **kwargs)


def accent_point(
    ax: "Axes",
    x: float,
    y: float,
    *,
    label: str | None = None,
    color: str | None = None,
    size: float = 30.0,
    zorder: int = 3,
    **kwargs,
) -> None:
    """Highlight a single point with the accent color.

    Optionally add a short text label offset slightly from the point.

    Args:
        ax: Matplotlib axes to modify.
        x: X coordinate of the point.
        y: Y coordinate of the point.
        label: Optional label text.
        color: Optional color override (defaults to accent color).
        size: Marker size.
        zorder: Z-order for the marker.
        **kwargs: Additional arguments passed to ax.scatter().
    """
    c = color or ACCENT
    ax.scatter([x], [y], s=size, color=c, zorder=zorder, **kwargs)

    if label:
        ax.text(
            x,
            y,
            label,
            ha="left",
            va="bottom",
            color=c,
            fontsize=plt.rcParams["font.size"],
        )


def event_line(
    ax: "Axes",
    x: float,
    *,
    text: str | None = None,
    y_text: float = 0.9,
    color: str | None = None,
    linewidth: float = 1.0,
    linestyle: str = "--",
    **kwargs,
) -> None:
    """Draw a vertical event marker with optional label.

    Args:
        ax: Matplotlib axes to modify.
        x: X coordinate for the vertical line.
        text: Optional label text.
        y_text: Y position for text (0-1 is fraction of y-span, otherwise data coords).
        color: Optional color override (defaults to accent color).
        linewidth: Line width.
        linestyle: Line style.
        **kwargs: Additional arguments passed to ax.axvline().
    """
    c = color or ACCENT
    ax.axvline(x, color=c, linewidth=linewidth, linestyle=linestyle, **kwargs)

    if text is not None:
        y0, y1 = ax.get_ylim()
        if 0.0 <= y_text <= 1.0:
            y = y0 + y_text * (y1 - y0)
        else:
            y = y_text

        ax.text(
            x,
            y,
            text,
            ha="left",
            va="center",
            color=c,
            fontsize=plt.rcParams["font.size"],
        )


def draw_waterfall(ax: "Axes", view: WaterfallView) -> None:
    """Draw a waterfall chart on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: WaterfallView containing the data to plot.
    """
    categories = np.asarray(view.categories)
    values = np.asarray(view.values)
    n = len(categories)

    # Calculate cumulative positions and bar positions/heights
    x_pos = np.arange(n)
    bottoms = np.zeros(n)
    heights = np.zeros(n)
    cumulative = 0.0

    for i in range(n):
        measure_type = view.measures[i] if view.measures is not None else None

        if measure_type == "absolute" or (measure_type is None and i == 0):
            # Absolute value: bar goes from 0 to value
            bottoms[i] = 0
            heights[i] = values[i]
            cumulative = values[i]
        elif measure_type == "total":
            # Total: bar goes from 0 to cumulative + value (which is the total)
            # For totals, the value in the array might be 0, so we calculate from previous cumulative
            bottoms[i] = 0
            heights[i] = cumulative + values[i] if values[i] != 0 else cumulative
            cumulative = heights[i]
        else:  # relative (default for middle values)
            # Relative: bar starts at previous cumulative, adds the value
            bottoms[i] = cumulative
            heights[i] = values[i]
            cumulative = cumulative + values[i]

    # Determine colors - vectorized
    if view.color:
        colors_list = [view.color] * n
    elif view.colors:
        colors_list = list(view.colors)
    else:
        # Default colors: green for positive, red for negative, blue for totals/absolute
        measures_arr = np.asarray(view.measures) if view.measures is not None else None
        is_total = (
            measures_arr == "total"
            if measures_arr is not None
            else np.zeros(n, dtype=bool)
        )
        is_absolute = (
            measures_arr == "absolute" if measures_arr is not None else None
        ) or (np.arange(n) == 0 if measures_arr is None else np.zeros(n, dtype=bool))
        is_positive = values >= 0

        colors_list = np.where(
            is_total | (np.arange(n) == 0 if measures_arr is None else is_absolute),
            "#1f77b4",  # blue
            np.where(is_positive, "#2ca02c", "#d62728"),  # green or red
        ).tolist()

    # Draw bars - vectorized
    ax.bar(
        x_pos,
        heights,
        bottom=bottoms,
        color=colors_list,
        edgecolor="black",
        linewidth=0.5,
    )

    # Set category labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, ha="right")


def draw_waffle(ax: "Axes", view: WaffleView) -> None:
    """Draw a waffle chart on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: WaffleView containing the data to plot.
    """
    categories = np.asarray(view.categories)
    values = np.asarray(view.values)
    total = np.sum(values)

    # Calculate grid size
    if view.rows is None and view.columns is None:
        # Default: try to make it roughly square
        grid_size = int(np.ceil(np.sqrt(total)))
        rows = grid_size
        columns = grid_size
    elif view.rows is not None:
        rows = view.rows
        columns = int(np.ceil(total / rows))
    elif view.columns is not None:
        columns = view.columns
        rows = int(np.ceil(total / columns))
    else:
        # Default: 10x10 grid
        rows = 10
        columns = 10

    # Create grid
    grid = np.zeros((rows, columns), dtype=int)
    category_map = {}

    # Fill grid with category indices
    idx = 0
    for cat_idx, val in enumerate(values):
        category_map[cat_idx] = categories[cat_idx]
        for _ in range(int(val)):
            if idx < rows * columns:
                row = idx // columns
                col = idx % columns
                grid[row, col] = cat_idx
                idx += 1

    # Determine colors
    if view.colors is not None:
        colors_list = list(view.colors)
    else:
        # Default blue gradient - vectorized
        n_cats = len(categories)
        if n_cats > 1:
            color_indices: np.ndarray = np.linspace(0.3, 1.0, n_cats)
        else:
            color_indices: list[float] = [0.65]
        cmap = cm.get_cmap("Blues")  # type: ignore[attr-defined]
        colors_list = [cmap(idx) for idx in color_indices]

    # Draw grid
    for row in range(rows):
        for col in range(columns):
            cat_idx = grid[row, col]
            color = colors_list[cat_idx] if cat_idx < len(colors_list) else "#cccccc"
            rect = plt.Rectangle(  # type: ignore[attr-defined]
                (col, rows - row - 1),
                1,
                1,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.add_patch(rect)

    ax.set_xlim(0, columns)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_dumbbell_range(
    ax: "Axes",
    categories: np.ndarray,
    values1: np.ndarray,
    values2: np.ndarray,
    colors: list[str] | np.ndarray | None,
    color: str | None,
    orientation: str,
    label1: str | None,
    linewidth: float,
) -> None:
    """Helper function to draw dumbbell/range charts (shared logic).

    Args:
        ax: Matplotlib axes to draw on.
        categories: Category labels.
        values1: First set of values.
        values2: Second set of values.
        colors: Optional list of colors.
        color: Optional single color.
        orientation: 'h' for horizontal or 'v' for vertical.
        label1: Optional label.
        linewidth: Line width.
    """
    n = len(categories)

    # Determine colors - vectorized
    if colors is not None:
        color_arr = np.asarray(colors)
        if len(color_arr) < n:
            color_arr = np.pad(
                color_arr,
                (0, n - len(color_arr)),
                constant_values=color_arr[-1] if len(color_arr) > 0 else "#1f77b4",
            )
        colors_list = color_arr[:n].tolist()
    elif color:
        colors_list = [color] * n
    else:
        colors_list = ["#1f77b4"] * n

    if orientation == "v":
        # Vertical orientation
        pos = np.arange(n)
        # Vectorized plotting
        for i, (v1, v2, pos_val, col) in enumerate(
            zip(values1, values2, pos, colors_list)
        ):
            ax.plot(
                [v1, v2], [pos_val, pos_val], color=col, linewidth=linewidth, zorder=1
            )
            ax.scatter(
                [v1, v2],
                [pos_val, pos_val],
                s=100,
                color=col,
                edgecolor="white",
                linewidth=1.5,
                zorder=2,
            )
        ax.set_yticks(pos)
        ax.set_yticklabels(categories)
        ax.set_xlabel(label1 or "Value")
    else:
        # Horizontal orientation (default)
        pos = np.arange(n)
        # Vectorized plotting
        for i, (v1, v2, pos_val, col) in enumerate(
            zip(values1, values2, pos, colors_list)
        ):
            ax.plot(
                [v1, v2], [pos_val, pos_val], color=col, linewidth=linewidth, zorder=1
            )
            ax.scatter(
                [v1, v2],
                [pos_val, pos_val],
                s=100,
                color=col,
                edgecolor="white",
                linewidth=1.5,
                zorder=2,
            )
        ax.set_yticks(pos)
        ax.set_yticklabels(categories)
        ax.set_xlabel(label1 or "Value")


def draw_dumbbell(ax: "Axes", view: DumbbellView) -> None:
    """Draw a dumbbell chart on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: DumbbellView containing the data to plot.
    """
    categories = np.asarray(view.categories)
    values1 = np.asarray(view.values1)
    values2 = np.asarray(view.values2)

    _draw_dumbbell_range(
        ax=ax,
        categories=categories,
        values1=values1,
        values2=values2,
        colors=view.colors,
        color=view.color,
        orientation=view.orientation,
        label1=view.label1,
        linewidth=2.0,
    )


def draw_range(ax: "Axes", view: RangeView) -> None:
    """Draw a range chart on the given axes (similar to dumbbell but different styling).

    Args:
        ax: Matplotlib axes to draw on.
        view: RangeView containing the data to plot.
    """
    categories = np.asarray(view.categories)
    values1 = np.asarray(view.values1)
    values2 = np.asarray(view.values2)

    _draw_dumbbell_range(
        ax=ax,
        categories=categories,
        values1=values1,
        values2=values2,
        colors=None,  # Range uses single color
        color=view.color or "#1f77b4",
        orientation=view.orientation,
        label1=view.label1,
        linewidth=3.0,  # Thicker line for range
    )


def draw_lollipop(ax: "Axes", view: LollipopView) -> None:
    """Draw a lollipop chart on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: LollipopView containing the data to plot.
    """
    categories = np.asarray(view.categories)
    values = np.asarray(view.values)
    n = len(categories)

    color = view.color or "#1f77b4"
    marker = view.marker or "o"
    linewidth = view.linewidth or 1.5

    pos = np.arange(n)
    if view.horizontal:
        # Horizontal lollipops - vectorized
        # Draw lines and markers vectorized
        for pos_val, val in zip(pos, values):
            ax.plot(
                [0, val], [pos_val, pos_val], color=color, linewidth=linewidth, zorder=1
            )
        ax.scatter(
            values,
            pos,
            s=100,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.5,
            zorder=2,
        )
        ax.set_yticks(pos)
        ax.set_yticklabels(categories)
        ax.axvline(x=0, color="black", linewidth=0.5, linestyle="-")
    else:
        # Vertical lollipops (default) - vectorized
        # Draw lines and markers vectorized
        for pos_val, val in zip(pos, values):
            ax.plot(
                [pos_val, pos_val], [0, val], color=color, linewidth=linewidth, zorder=1
            )
        ax.scatter(
            pos,
            values,
            s=100,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.5,
            zorder=2,
        )
        ax.set_xticks(pos)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")


def draw_slope(ax: "Axes", view: SlopeView) -> None:
    """Draw a slope chart on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: SlopeView containing the data to plot.
    """
    x = np.asarray(view.x)
    if len(x) != 2:
        raise ValueError("Slope chart requires exactly 2 x values")

    # Default colors
    default_colors = np.array(
        [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    )

    for i, (group_name, y_values) in enumerate(view.groups.items()):
        y = np.asarray(y_values)
        if len(y) != 2:
            raise ValueError(f"Group '{group_name}' must have exactly 2 y values")

        # Determine color - use dict.get for cleaner code
        color = (view.colors or {}).get(
            group_name, default_colors[i % len(default_colors)]
        )

        # Draw line
        ax.plot(
            x,
            y,
            color=color,
            linewidth=2,
            marker="o",
            markersize=8,
            label=group_name,
            zorder=2,
        )

    # Calculate xlim with vectorized operations
    x_range = x[1] - x[0]
    ax.set_xlim(x[0] - 0.1 * x_range, x[1] + 0.1 * x_range)


def draw_metric(ax: "Axes", view: MetricView) -> None:
    """Draw a metric display on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: MetricView containing the data to display.
    """
    # Clear axes and remove spines
    ax.axis("off")

    # Format value
    value_str = (
        f"{view.value:,.0f}"
        if isinstance(view.value, (int, float))
        else str(view.value)
    )
    if view.prefix:
        value_str = view.prefix + value_str
    if view.suffix:
        value_str = value_str + view.suffix

    # Format delta
    delta_str = ""
    if view.delta is not None:
        delta_val = abs(view.delta)
        delta_str = (
            f"{delta_val:+,.0f}"
            if isinstance(delta_val, (int, float))
            else str(delta_val)
        )
        if view.suffix:
            delta_str = delta_str + view.suffix

    # Determine colors
    value_color = view.value_color if view.value_color else "black"
    if view.delta is not None:
        if view.delta_color:
            delta_color = view.delta_color
        else:
            delta_color = "#2ca02c" if view.delta >= 0 else "#d62728"
    else:
        delta_color = "black"

    # Display title
    ax.text(
        0.5,
        0.8,
        view.title,
        ha="center",
        va="top",
        fontsize=14,
        color="gray",
        transform=ax.transAxes,
    )

    # Display value
    ax.text(
        0.5,
        0.5,
        value_str,
        ha="center",
        va="center",
        fontsize=32,
        fontweight="bold",
        color=value_color,
        transform=ax.transAxes,
    )

    # Display delta
    if delta_str:
        ax.text(
            0.5,
            0.2,
            delta_str,
            ha="center",
            va="bottom",
            fontsize=18,
            color=delta_color,
            transform=ax.transAxes,
        )


def draw_box(ax: "Axes", view: BoxView) -> None:
    """Draw a box plot on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: BoxView containing the data to plot.
    """
    bp = ax.boxplot(
        view.data,
        labels=view.labels,  # type: ignore[call-overload,arg-type]
        positions=view.positions,
        patch_artist=True,
        showmeans=view.show_means,
        showfliers=view.show_outliers,
    )

    # Set colors - vectorized
    boxes = bp["boxes"]
    if view.color:
        for patch in boxes:
            patch.set_facecolor(view.color)
    elif view.colors:
        colors_arr = np.asarray(view.colors)
        for i, patch in enumerate(boxes):
            if i < len(colors_arr):
                patch.set_facecolor(colors_arr[i])

    # Style boxes - vectorized operations
    for patch in boxes:
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
        patch.set_alpha(0.7)

    # Style other elements - more Pythonic
    style_elements = ["whiskers", "fliers", "means", "medians", "caps"]
    for element in style_elements:
        if element in bp:
            for item in bp[element]:
                item.set_color("black")
                item.set_linewidth(1.0)


def draw_violin(ax: "Axes", view: ViolinView) -> None:
    """Draw a violin plot on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: ViolinView containing the data to plot.
    """
    parts = ax.violinplot(
        view.data,
        positions=view.positions,
        showmeans=view.show_means,
        showmedians=view.show_medians,
    )

    # Set colors - vectorized
    bodies = parts["bodies"]
    if view.color:
        for pc in bodies:  # type: ignore[union-attr]
            pc.set_facecolor(view.color)  # type: ignore[attr-defined]
            pc.set_alpha(0.7)  # type: ignore[attr-defined]
    elif view.colors:
        colors_arr = np.asarray(view.colors)
        for i, pc in enumerate(bodies):  # type: ignore[union-attr,arg-type]
            if i < len(colors_arr):
                pc.set_facecolor(colors_arr[i])  # type: ignore[attr-defined,union-attr]
                pc.set_alpha(0.7)  # type: ignore[attr-defined,union-attr]

    # Style violins
    for pc in bodies:  # type: ignore[union-attr]
        pc.set_edgecolor("black")  # type: ignore[attr-defined]
        pc.set_linewidth(1.0)  # type: ignore[attr-defined]

    # Style other elements - more Pythonic
    style_elements = ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]
    for element in style_elements:
        if element in parts:
            parts[element].set_color("black")
            parts[element].set_linewidth(1.0)

    # Set labels if provided
    if view.labels is not None:
        positions = (
            view.positions
            if view.positions is not None
            else np.arange(1, len(view.data) + 1)
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(view.labels)
