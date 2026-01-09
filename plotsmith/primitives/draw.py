"""Drawing functions that accept only Layer 1 objects."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from plotsmith.objects.views import (
    BandView,
    BarView,
    FigureSpec,
    HeatmapView,
    HistogramView,
    ScatterView,
    SeriesView,
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
    kwargs = {}
    if view.marker is not None:
        kwargs["marker"] = view.marker
    if view.linewidth is not None:
        kwargs["linewidth"] = view.linewidth
    if view.alpha is not None:
        kwargs["alpha"] = view.alpha

    ax.plot(view.x, view.y, label=view.label, **kwargs)


def draw_band(ax: "Axes", view: BandView) -> None:
    """Draw a confidence band or shaded region on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: BandView containing the data to plot.
    """
    kwargs = {}
    if view.alpha is not None:
        kwargs["alpha"] = view.alpha

    ax.fill_between(view.x, view.y_lower, view.y_upper, label=view.label, **kwargs)


def draw_scatter(ax: "Axes", view: ScatterView) -> None:
    """Draw a scatter plot on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: ScatterView containing the data to plot.
    """
    kwargs = {}
    if view.marker is not None:
        kwargs["marker"] = view.marker
    if view.alpha is not None:
        kwargs["alpha"] = view.alpha
    if view.s is not None:
        kwargs["s"] = view.s

    ax.scatter(view.x, view.y, label=view.label, **kwargs)


def draw_histogram(ax: "Axes", view: HistogramView) -> None:
    """Draw a histogram on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: HistogramView containing the data to plot.
    """
    kwargs = {}
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

    ax.hist(view.values, **kwargs)


def draw_bar(ax: "Axes", view: BarView) -> None:
    """Draw a bar chart on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: BarView containing the data to plot.
    """
    kwargs = {}
    if view.color is not None:
        kwargs["color"] = view.color
    if view.edgecolor is not None:
        kwargs["edgecolor"] = view.edgecolor
    if view.alpha is not None:
        kwargs["alpha"] = view.alpha
    if view.label is not None:
        kwargs["label"] = view.label

    if view.horizontal:
        ax.barh(view.x, view.height, **kwargs)
    else:
        ax.bar(view.x, view.height, **kwargs)


def draw_heatmap(ax: "Axes", view: HeatmapView) -> None:
    """Draw a heatmap on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        view: HeatmapView containing the data to plot.
    """
    kwargs = {}
    if view.cmap is not None:
        kwargs["cmap"] = view.cmap
    if view.vmin is not None:
        kwargs["vmin"] = view.vmin
    if view.vmax is not None:
        kwargs["vmax"] = view.vmax

    im = ax.imshow(view.data, aspect="auto", **kwargs)

    # Set labels if provided
    if view.x_labels is not None:
        ax.set_xticks(range(len(view.x_labels)))
        ax.set_xticklabels(view.x_labels)
    if view.y_labels is not None:
        ax.set_yticks(range(len(view.y_labels)))
        ax.set_yticklabels(view.y_labels)

    # Add annotations if requested
    if view.annotate:
        fmt = view.fmt or ".2f"
        for i in range(view.data.shape[0]):
            for j in range(view.data.shape[1]):
                text = ax.text(
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
        x_min, x_max = float(min(x_data)), float(max(x_data))
        ax.spines["bottom"].set_bounds(x_min, x_max)

    if y_data is not None:
        y_min, y_max = float(min(y_data)), float(max(y_data))
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
        collection.set_edgecolors("#000000")
        collection.set_facecolors("#FFFFFF")
        collection.set_linewidths(1.0)


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


def emphasize_last(ax: "Axes", x: float, y: float, *, size: float = 30.0, **kwargs) -> None:
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

