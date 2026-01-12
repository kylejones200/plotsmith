"""Extended tests for Layer 2: primitives - all new draw functions."""

# Use Agg backend for testing (non-interactive)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotsmith.objects.views import (
    BoxView,
    DumbbellView,
    LollipopView,
    MetricView,
    RangeView,
    SlopeView,
    ViolinView,
    WaffleView,
    WaterfallView,
)
from plotsmith.primitives.draw import (
    draw_box,
    draw_dumbbell,
    draw_lollipop,
    draw_metric,
    draw_range,
    draw_slope,
    draw_violin,
    draw_waffle,
    draw_waterfall,
)


class TestDrawWaterfall:
    """Tests for draw_waterfall function."""

    def test_draw_waterfall_basic(self):
        """Test that draw_waterfall runs without error."""
        fig, ax = plt.subplots()
        view = WaterfallView(
            categories=np.array(["A", "B", "C"]), values=np.array([10, 20, 15])
        )
        draw_waterfall(ax, view)  # Should not raise
        plt.close(fig)


class TestDrawWaffle:
    """Tests for draw_waffle function."""

    def test_draw_waffle_basic(self):
        """Test that draw_waffle runs without error."""
        fig, ax = plt.subplots()
        view = WaffleView(
            categories=np.array(["A", "B"]), values=np.array([10, 20])
        )
        draw_waffle(ax, view)  # Should not raise
        plt.close(fig)


class TestDrawDumbbell:
    """Tests for draw_dumbbell function."""

    def test_draw_dumbbell_basic(self):
        """Test that draw_dumbbell runs without error."""
        fig, ax = plt.subplots()
        view = DumbbellView(
            categories=np.array(["A", "B"]),
            values1=np.array([10, 20]),
            values2=np.array([15, 25]),
        )
        draw_dumbbell(ax, view)  # Should not raise
        plt.close(fig)


class TestDrawRange:
    """Tests for draw_range function."""

    def test_draw_range_basic(self):
        """Test that draw_range runs without error."""
        fig, ax = plt.subplots()
        view = RangeView(
            categories=np.array(["A", "B"]),
            values1=np.array([10, 20]),
            values2=np.array([15, 25]),
        )
        draw_range(ax, view)  # Should not raise
        plt.close(fig)


class TestDrawLollipop:
    """Tests for draw_lollipop function."""

    def test_draw_lollipop_basic(self):
        """Test that draw_lollipop runs without error."""
        fig, ax = plt.subplots()
        view = LollipopView(
            categories=np.array(["A", "B", "C"]), values=np.array([10, 25, 15])
        )
        draw_lollipop(ax, view)  # Should not raise
        plt.close(fig)


class TestDrawSlope:
    """Tests for draw_slope function."""

    def test_draw_slope_basic(self):
        """Test that draw_slope runs without error."""
        fig, ax = plt.subplots()
        view = SlopeView(
            x=np.array([2020, 2021]),
            groups={"USA": np.array([10, 12]), "Canada": np.array([12, 15])},
        )
        draw_slope(ax, view)  # Should not raise
        plt.close(fig)


class TestDrawMetric:
    """Tests for draw_metric function."""

    def test_draw_metric_basic(self):
        """Test that draw_metric runs without error."""
        fig, ax = plt.subplots()
        view = MetricView(title="Sales", value=12345, delta=123)
        draw_metric(ax, view)  # Should not raise
        plt.close(fig)


class TestDrawBox:
    """Tests for draw_box function."""

    def test_draw_box_basic(self):
        """Test that draw_box runs without error."""
        fig, ax = plt.subplots()
        view = BoxView(data=[np.array([1, 2, 3]), np.array([4, 5, 6])])
        draw_box(ax, view)  # Should not raise
        plt.close(fig)


class TestDrawViolin:
    """Tests for draw_violin function."""

    def test_draw_violin_basic(self):
        """Test that draw_violin runs without error."""
        fig, ax = plt.subplots()
        view = ViolinView(data=[np.array([1, 2, 3]), np.array([4, 5, 6])])
        draw_violin(ax, view)  # Should not raise
        plt.close(fig)

