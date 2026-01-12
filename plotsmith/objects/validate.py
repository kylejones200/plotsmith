"""Validators for checking shapes and alignment of view objects."""

from collections.abc import Sequence

import numpy as np

from plotsmith.exceptions import ValidationError
from plotsmith.objects.views import (
    BandView,
    BarView,
    BoxView,
    DumbbellView,
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

# Union type for all possible view types
View = (
    SeriesView
    | BandView
    | ScatterView
    | HistogramView
    | BarView
    | HeatmapView
    | WaterfallView
    | WaffleView
    | DumbbellView
    | RangeView
    | LollipopView
    | SlopeView
    | BoxView
    | ViolinView
    | MetricView,
)


def _validate_matching_lengths(
    *arrays: np.ndarray | list,
    names: list[str],
    view_type: str,
) -> None:
    """Helper to validate multiple arrays have matching lengths.

    Args:
        *arrays: Variable number of arrays to validate.
        names: Names for each array (for error messages).
        view_type: Type of view being validated (for error messages).

    Raises:
        ValidationError: If arrays have mismatched lengths.
    """
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        length_info = ", ".join(
            f"{name}: {length}" for name, length in zip(names, lengths)
        )
        raise ValidationError(
            f"{view_type} {', '.join(names)} must have same length. Got: {length_info}",
            context={
                "view_type": view_type,
                "lengths": dict(zip(names, lengths)),
            },
        )


def validate_series_view(view: SeriesView) -> None:
    """Validate that a SeriesView has matching x and y lengths.

    Args:
        view: The SeriesView to validate.

    Raises:
        ValidationError: If x and y arrays have mismatched lengths.
    """
    _validate_matching_lengths(view.x, view.y, names=["x", "y"], view_type="SeriesView")


def validate_band_view(view: BandView) -> None:
    """Validate that a BandView has matching x, y_lower, and y_upper lengths.

    Args:
        view: The BandView to validate.

    Raises:
        ValidationError: If arrays have mismatched lengths.
    """
    _validate_matching_lengths(
        view.x,
        view.y_lower,
        view.y_upper,
        names=["x", "y_lower", "y_upper"],
        view_type="BandView",
    )


def validate_scatter_view(view: ScatterView) -> None:
    """Validate that a ScatterView has matching x and y lengths.

    Args:
        view: The ScatterView to validate.

    Raises:
        ValidationError: If x and y arrays have mismatched lengths.
    """
    _validate_matching_lengths(
        view.x, view.y, names=["x", "y"], view_type="ScatterView"
    )


def validate_bar_view(view: BarView) -> None:
    """Validate that a BarView has matching x and height lengths.

    Args:
        view: The BarView to validate.

    Raises:
        ValidationError: If x and height arrays have mismatched lengths.
    """
    x_length = len(view.x) if isinstance(view.x, (np.ndarray, list)) else 0
    height_length = len(view.height)

    if x_length != height_length:
        raise ValidationError(
            f"BarView x and height must have same length, got {x_length} and {height_length}",
            context={
                "view_type": "BarView",
                "x_length": x_length,
                "height_length": height_length,
                "x_type": type(view.x).__name__,
            },
        )


def validate_waterfall_view(view: WaterfallView) -> None:
    """Validate that a WaterfallView has matching categories and values lengths.

    Args:
        view: The WaterfallView to validate.

    Raises:
        ValidationError: If arrays have mismatched lengths.
    """
    categories = np.asarray(view.categories)
    values = np.asarray(view.values)
    _validate_matching_lengths(
        categories,
        values,
        names=["categories", "values"],
        view_type="WaterfallView",
    )
    if view.measures is not None:
        measures = np.asarray(view.measures)
        _validate_matching_lengths(
            measures,
            values,
            names=["measures", "values"],
            view_type="WaterfallView",
        )


def validate_dumbbell_view(view: DumbbellView) -> None:
    """Validate that a DumbbellView has matching lengths.

    Args:
        view: The DumbbellView to validate.

    Raises:
        ValidationError: If arrays have mismatched lengths.
    """
    categories = np.asarray(view.categories)
    values1 = np.asarray(view.values1)
    values2 = np.asarray(view.values2)
    _validate_matching_lengths(
        categories,
        values1,
        values2,
        names=["categories", "values1", "values2"],
        view_type="DumbbellView",
    )


def validate_range_view(view: RangeView) -> None:
    """Validate that a RangeView has matching lengths.

    Args:
        view: The RangeView to validate.

    Raises:
        ValidationError: If arrays have mismatched lengths.
    """
    categories = np.asarray(view.categories)
    values1 = np.asarray(view.values1)
    values2 = np.asarray(view.values2)
    _validate_matching_lengths(
        categories,
        values1,
        values2,
        names=["categories", "values1", "values2"],
        view_type="RangeView",
    )


def validate_lollipop_view(view: LollipopView) -> None:
    """Validate that a LollipopView has matching categories and values lengths.

    Args:
        view: The LollipopView to validate.

    Raises:
        ValidationError: If arrays have mismatched lengths.
    """
    categories = np.asarray(view.categories)
    values = np.asarray(view.values)
    _validate_matching_lengths(
        categories,
        values,
        names=["categories", "values"],
        view_type="LollipopView",
    )


def validate_slope_view(view: SlopeView) -> None:
    """Validate that a SlopeView has correct structure.

    Args:
        view: The SlopeView to validate.

    Raises:
        ValueError: If structure is invalid.
    """
    x = np.asarray(view.x)
    if len(x) != 2:
        raise ValidationError(
            f"SlopeView x must have exactly 2 values, got {len(x)}",
            context={"view_type": "SlopeView", "x_length": len(x)},
        )
    for group_name, y_values in view.groups.items():
        y = np.asarray(y_values)
        if len(y) != 2:
            raise ValidationError(
                f"SlopeView group '{group_name}' must have exactly 2 y values, got {len(y)}",
                context={
                    "view_type": "SlopeView",
                    "group_name": group_name,
                    "y_length": len(y),
                },
            )


def validate_waffle_view(view: WaffleView) -> None:
    """Validate that a WaffleView has matching categories and values lengths.

    Args:
        view: The WaffleView to validate.

    Raises:
        ValidationError: If arrays have mismatched lengths.
    """
    categories = np.asarray(view.categories)
    values = np.asarray(view.values)
    _validate_matching_lengths(
        categories,
        values,
        names=["categories", "values"],
        view_type="WaffleView",
    )
    if np.any(values < 0):
        raise ValidationError(
            "WaffleView values must be non-negative",
            context={
                "view_type": "WaffleView",
                "negative_indices": np.where(values < 0)[0].tolist(),
            },
        )


def validate_box_view(view: BoxView) -> None:
    """Validate that a BoxView has valid data.

    Args:
        view: The BoxView to validate.

    Raises:
        ValueError: If data is invalid.
    """
    if not view.data:
        raise ValidationError(
            "BoxView data must not be empty", context={"view_type": "BoxView"}
        )
    if view.labels is not None and len(view.labels) != len(view.data):
        raise ValidationError(
            f"BoxView labels must have same length as data, got {len(view.labels)} and {len(view.data)}",
            context={
                "view_type": "BoxView",
                "labels_length": len(view.labels),
                "data_length": len(view.data),
            },
        )


def validate_violin_view(view: ViolinView) -> None:
    """Validate that a ViolinView has valid data.

    Args:
        view: The ViolinView to validate.

    Raises:
        ValueError: If data is invalid.
    """
    if not view.data:
        raise ValidationError(
            "ViolinView data must not be empty", context={"view_type": "ViolinView"}
        )
    if view.labels is not None and len(view.labels) != len(view.data):
        raise ValidationError(
            f"ViolinView labels must have same length as data, got {len(view.labels)} and {len(view.data)}",
            context={
                "view_type": "ViolinView",
                "labels_length": len(view.labels),
                "data_length": len(view.data),
            },
        )


def validate_all_views(views: Sequence[View]) -> None:
    """Validate a list of views.

    Args:
        views: Sequence of view objects to validate. Accepts any sequence
            containing one or more view types (SeriesView, BandView, etc.).

    Raises:
        ValidationError: If any view fails validation.
        TypeError: If view type is unknown.
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
            # HeatmapView validation
            if view.data.ndim != 2:
                raise ValidationError(
                    f"HeatmapView data must be 2D, got {view.data.ndim}D",
                    context={
                        "view_type": "HeatmapView",
                        "data_shape": view.data.shape,
                        "data_ndim": view.data.ndim,
                    },
                )
        elif isinstance(view, WaterfallView):
            validate_waterfall_view(view)
        elif isinstance(view, WaffleView):
            validate_waffle_view(view)
        elif isinstance(view, DumbbellView):
            validate_dumbbell_view(view)
        elif isinstance(view, RangeView):
            validate_range_view(view)
        elif isinstance(view, LollipopView):
            validate_lollipop_view(view)
        elif isinstance(view, SlopeView):
            validate_slope_view(view)
        elif isinstance(view, BoxView):
            validate_box_view(view)
        elif isinstance(view, ViolinView):
            validate_violin_view(view)
        elif isinstance(view, MetricView):
            # MetricView doesn't need validation (just displays a value)
            pass
        else:
            raise TypeError(f"Unknown view type: {type(view)}")
