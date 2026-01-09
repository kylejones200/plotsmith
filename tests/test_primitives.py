"""Tests for Layer 2: primitives."""

import numpy as np
import pytest

# Use Agg backend for testing (non-interactive)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotsmith.objects.views import BandView, FigureSpec, ScatterView, SeriesView
from plotsmith.primitives.draw import (
    apply_axes_style,
    draw_band,
    draw_scatter,
    draw_series,
    minimal_axes,
)


class TestDrawSeries:
    """Tests for draw_series function."""

    def test_draw_series_basic(self):
        """Test that draw_series runs without error."""
        fig, ax = plt.subplots()
        view = SeriesView(x=np.array([1, 2, 3]), y=np.array([10, 20, 30]))
        draw_series(ax, view)  # Should not raise
        plt.close(fig)

    def test_draw_series_with_style(self):
        """Test that draw_series applies style hints."""
        fig, ax = plt.subplots()
        view = SeriesView(
            x=np.array([1, 2, 3]),
            y=np.array([10, 20, 30]),
            marker="o",
            linewidth=2.0,
            alpha=0.5,
            label="test",
        )
        draw_series(ax, view)  # Should not raise
        plt.close(fig)


class TestDrawBand:
    """Tests for draw_band function."""

    def test_draw_band_basic(self):
        """Test that draw_band runs without error."""
        fig, ax = plt.subplots()
        view = BandView(
            x=np.array([1, 2, 3]),
            y_lower=np.array([5, 10, 15]),
            y_upper=np.array([15, 20, 25]),
        )
        draw_band(ax, view)  # Should not raise
        plt.close(fig)

    def test_draw_band_with_style(self):
        """Test that draw_band applies style hints."""
        fig, ax = plt.subplots()
        view = BandView(
            x=np.array([1, 2, 3]),
            y_lower=np.array([5, 10, 15]),
            y_upper=np.array([15, 20, 25]),
            alpha=0.3,
            label="confidence",
        )
        draw_band(ax, view)  # Should not raise
        plt.close(fig)


class TestDrawScatter:
    """Tests for draw_scatter function."""

    def test_draw_scatter_basic(self):
        """Test that draw_scatter runs without error."""
        fig, ax = plt.subplots()
        view = ScatterView(x=np.array([1, 2, 3]), y=np.array([10, 20, 30]))
        draw_scatter(ax, view)  # Should not raise
        plt.close(fig)

    def test_draw_scatter_with_style(self):
        """Test that draw_scatter applies style hints."""
        fig, ax = plt.subplots()
        view = ScatterView(
            x=np.array([1, 2, 3]),
            y=np.array([10, 20, 30]),
            marker="s",
            alpha=0.7,
            s=50.0,
            label="points",
        )
        draw_scatter(ax, view)  # Should not raise
        plt.close(fig)


class TestMinimalAxes:
    """Tests for minimal_axes function."""

    def test_minimal_axes_applies_styling(self):
        """Test that minimal_axes applies minimalist styling."""
        fig, ax = plt.subplots()
        minimal_axes(ax)

        # Check that top and right spines are hidden
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        # Check that left and bottom spines are visible
        assert ax.spines["left"].get_visible()
        assert ax.spines["bottom"].get_visible()

        plt.close(fig)


class TestApplyAxesStyle:
    """Tests for apply_axes_style function."""

    def test_apply_axes_style_basic(self):
        """Test that apply_axes_style runs without error."""
        fig, ax = plt.subplots()
        spec = FigureSpec(title="Test", xlabel="X", ylabel="Y")
        apply_axes_style(ax, spec)

        assert ax.get_title() == "Test"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        plt.close(fig)

    def test_apply_axes_style_with_limits(self):
        """Test that apply_axes_style applies axis limits."""
        fig, ax = plt.subplots()
        spec = FigureSpec(xlim=(0, 10), ylim=(0, 20))
        apply_axes_style(ax, spec)

        assert ax.get_xlim() == (0, 10)
        assert ax.get_ylim() == (0, 20)
        plt.close(fig)

