"""PlotSmith: A layered plotting library for ML models."""

# Public API: Only workflows and view objects
from plotsmith.objects import (
    BandView,
    BarView,
    FigureSpec,
    HeatmapView,
    HistogramView,
    ScatterView,
    SeriesView,
)
from plotsmith.primitives.draw import (
    ACCENT,
    accent_point,
    add_range_frame,
    direct_label,
    emphasize_last,
    event_line,
    force_bar_zero,
    minimal_axes,
    note,
    style_bar_plot,
    style_line_plot,
    style_scatter_plot,
    tidy_axes,
)
from plotsmith.workflows import (
    figure,
    plot_backtest,
    plot_bar,
    plot_heatmap,
    plot_histogram,
    plot_model_comparison,
    plot_residuals,
    plot_timeseries,
    small_multiples,
)

__all__ = [
    # View objects
    "BandView",
    "BarView",
    "FigureSpec",
    "HeatmapView",
    "HistogramView",
    "ScatterView",
    "SeriesView",
    # Workflows
    "figure",
    "plot_backtest",
    "plot_bar",
    "plot_heatmap",
    "plot_histogram",
    "plot_model_comparison",
    "plot_residuals",
    "plot_timeseries",
    "small_multiples",
    # Styling helpers
    "ACCENT",
    "accent_point",
    "add_range_frame",
    "direct_label",
    "emphasize_last",
    "event_line",
    "force_bar_zero",
    "minimal_axes",
    "note",
    "style_bar_plot",
    "style_line_plot",
    "style_scatter_plot",
    "tidy_axes",
]

