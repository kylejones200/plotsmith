"""Validators for checking shapes and alignment of view objects."""

import numpy as np

from plotsmith.objects.views import (
    BandView,
    BarView,
    HeatmapView,
    HistogramView,
    ScatterView,
    SeriesView,
)


def validate_series_view(view: SeriesView) -> None:
    """Validate that a SeriesView has matching x and y lengths.

    Args:
        view: The SeriesView to validate.

    Raises:
        ValueError: If x and y arrays have mismatched lengths.
    """
    if len(view.x) != len(view.y):
        raise ValueError(
            f"SeriesView x and y must have same length, got {len(view.x)} and {len(view.y)}"
        )


def validate_band_view(view: BandView) -> None:
    """Validate that a BandView has matching x, y_lower, and y_upper lengths.

    Args:
        view: The BandView to validate.

    Raises:
        ValueError: If arrays have mismatched lengths.
    """
    if len(view.x) != len(view.y_lower):
        raise ValueError(
            f"BandView x and y_lower must have same length, got {len(view.x)} and {len(view.y_lower)}"
        )
    if len(view.x) != len(view.y_upper):
        raise ValueError(
            f"BandView x and y_upper must have same length, got {len(view.x)} and {len(view.y_upper)}"
        )
    if len(view.y_lower) != len(view.y_upper):
        raise ValueError(
            f"BandView y_lower and y_upper must have same length, got {len(view.y_lower)} and {len(view.y_upper)}"
        )


def validate_scatter_view(view: ScatterView) -> None:
    """Validate that a ScatterView has matching x and y lengths.

    Args:
        view: The ScatterView to validate.

    Raises:
        ValueError: If x and y arrays have mismatched lengths.
    """
    if len(view.x) != len(view.y):
        raise ValueError(
            f"ScatterView x and y must have same length, got {len(view.x)} and {len(view.y)}"
        )


def validate_bar_view(view: BarView) -> None:
    """Validate that a BarView has matching x and height lengths.

    Args:
        view: The BarView to validate.

    Raises:
        ValueError: If x and height arrays have mismatched lengths.
    """
    if isinstance(view.x, np.ndarray):
        if len(view.x) != len(view.height):
            raise ValueError(
                f"BarView x and height must have same length, got {len(view.x)} and {len(view.height)}"
            )
    elif isinstance(view.x, list):
        if len(view.x) != len(view.height):
            raise ValueError(
                f"BarView x and height must have same length, got {len(view.x)} and {len(view.height)}"
            )


def validate_all_views(
    views: list[SeriesView | BandView | ScatterView | HistogramView | BarView | HeatmapView],
) -> None:
    """Validate a list of views.

    Args:
        views: List of view objects to validate.

    Raises:
        ValueError: If any view fails validation.
    """
    for view in views:
        if isinstance(view, SeriesView):
            validate_series_view(view)
        elif isinstance(view, BandView):
            validate_band_view(view)
        elif isinstance(view, ScatterView):
            validate_scatter_view(view)
        elif isinstance(view, BarView):
            validate_bar_view(view)
        elif isinstance(view, HistogramView):
            # HistogramView doesn't need length validation
            pass
        elif isinstance(view, HeatmapView):
            # HeatmapView doesn't need length validation (2D array)
            if view.data.ndim != 2:
                raise ValueError(f"HeatmapView data must be 2D, got {view.data.ndim}D")
        else:
            raise TypeError(f"Unknown view type: {type(view)}")

