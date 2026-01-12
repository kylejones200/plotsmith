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
    RangeView,
    ScatterView,
    SeriesView,
    SlopeView,
    ViolinView,
    WaffleView,
    WaterfallView,
)


def validate_series_view(view: SeriesView) -> None:
    """Validate that a SeriesView has matching x and y lengths.

    Args:
        view: The SeriesView to validate.

    Raises:
        ValidationError: If x and y arrays have mismatched lengths.
    """
    if len(view.x) != len(view.y):
        raise ValidationError(
            f"SeriesView x and y must have same length, got {len(view.x)} and {len(view.y)}",
            context={
                "view_type": "SeriesView",
                "x_length": len(view.x),
                "y_length": len(view.y),
            },
        )


def validate_band_view(view: BandView) -> None:
    """Validate that a BandView has matching x, y_lower, and y_upper lengths.

    Args:
        view: The BandView to validate.

    Raises:
        ValidationError: If arrays have mismatched lengths.
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
        ValidationError: If x and y arrays have mismatched lengths.
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


def validate_waterfall_view(view: WaterfallView) -> None:
    """Validate that a WaterfallView has matching categories and values lengths.

    Args:
        view: The WaterfallView to validate.

    Raises:
        ValidationError: If arrays have mismatched lengths.
    """
    categories = np.asarray(view.categories)
    values = np.asarray(view.values)
    if len(categories) != len(values):
        raise ValidationError(
            f"WaterfallView categories and values must have same length, got {len(categories)} and {len(values)}",
            context={
                "view_type": "WaterfallView",
                "categories_length": len(categories),
                "values_length": len(values),
            },
        )
    if view.measures is not None:
        measures = np.asarray(view.measures)
        if len(measures) != len(values):
            raise ValidationError(
                f"WaterfallView measures must have same length as values, got {len(measures)} and {len(values)}",
                context={
                    "view_type": "WaterfallView",
                    "measures_length": len(measures),
                    "values_length": len(values),
                },
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
    if len(categories) != len(values1) or len(categories) != len(values2):
        raise ValidationError(
            f"DumbbellView categories, values1, and values2 must have same length, "
            f"got {len(categories)}, {len(values1)}, {len(values2)}",
            context={
                "view_type": "DumbbellView",
                "categories_length": len(categories),
                "values1_length": len(values1),
                "values2_length": len(values2),
            },
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
    if len(categories) != len(values1) or len(categories) != len(values2):
        raise ValidationError(
            f"RangeView categories, values1, and values2 must have same length, "
            f"got {len(categories)}, {len(values1)}, {len(values2)}",
            context={
                "view_type": "RangeView",
                "categories_length": len(categories),
                "values1_length": len(values1),
                "values2_length": len(values2),
            },
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
    if len(categories) != len(values):
        raise ValidationError(
            f"LollipopView categories and values must have same length, got {len(categories)} and {len(values)}",
            context={
                "view_type": "LollipopView",
                "categories_length": len(categories),
                "values_length": len(values),
            },
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
    if len(categories) != len(values):
        raise ValidationError(
            f"WaffleView categories and values must have same length, got {len(categories)} and {len(values)}",
            context={
                "view_type": "WaffleView",
                "categories_length": len(categories),
                "values_length": len(values),
            },
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


def validate_all_views(
    views: Sequence[
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
    ],
) -> None:
    """Validate a list of views.

    Args:
        views: List of view objects to validate.

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
        else:
            raise TypeError(f"Unknown view type: {type(view)}")
