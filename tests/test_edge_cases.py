"""Edge case tests for PlotSmith - empty data, NaN, invalid inputs."""

# Use Agg backend for testing (non-interactive)
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from plotsmith.exceptions import ValidationError
from plotsmith.workflows.workflows import (
    plot_bar,
    plot_box,
    plot_histogram,
    plot_scatter,
    plot_timeseries,
    plot_violin,
)


class TestEmptyData:
    """Tests for handling empty data."""

    def test_plot_timeseries_empty_series(self):
        """Test that plot_timeseries handles empty Series."""
        data = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        fig, ax = plot_timeseries(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_histogram_empty_array(self):
        """Test that plot_histogram handles empty array."""
        data = np.array([])
        fig, ax = plot_histogram(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_bar_empty_categories(self):
        """Test that plot_bar handles empty categories."""
        categories = []
        values = []
        fig, ax = plot_bar(categories, values)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestNaNHandling:
    """Tests for handling NaN values."""

    def test_plot_timeseries_with_nan(self):
        """Test that plot_timeseries handles NaN values."""
        data = pd.Series([1, np.nan, 3, 4, 5], index=pd.date_range("2020-01-01", periods=5))
        fig, ax = plot_timeseries(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_scatter_with_nan(self):
        """Test that plot_scatter handles NaN values."""
        df = pd.DataFrame({"x": [1, np.nan, 3], "y": [10, 20, 30]})
        fig, ax = plot_scatter(df, x="x", y="y")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_histogram_with_nan(self):
        """Test that plot_histogram handles NaN values."""
        data = np.array([1, 2, np.nan, 4, 5])
        fig, ax = plot_histogram(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestInvalidInputs:
    """Tests for invalid input handling."""

    def test_plot_timeseries_invalid_type(self):
        """Test that plot_timeseries raises error for invalid type."""
        with pytest.raises((TypeError, ValueError)):
            plot_timeseries([1, 2, 3])  # Not Series or DataFrame

    def test_plot_scatter_missing_column(self):
        """Test that plot_scatter raises error for missing column."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises((ValueError, KeyError)):
            plot_scatter(df, x="x", y="missing_column")

    def test_plot_box_missing_column(self):
        """Test that plot_box raises error for missing column."""
        df = pd.DataFrame({"category": ["A", "B"]})
        with pytest.raises((ValueError, KeyError)):
            plot_box(df, x="category", y="missing_value")

    def test_plot_bar_mismatched_lengths(self):
        """Test that plot_bar handles mismatched lengths."""
        categories = ["A", "B", "C"]
        values = [10, 20]  # Mismatched length
        with pytest.raises((ValueError, ValidationError)):
            plot_bar(categories, values)


class TestSingleValueData:
    """Tests for single value data."""

    def test_plot_timeseries_single_value(self):
        """Test that plot_timeseries handles single value."""
        data = pd.Series([1], index=pd.date_range("2020-01-01", periods=1))
        fig, ax = plot_timeseries(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_histogram_single_value(self):
        """Test that plot_histogram handles single value."""
        data = np.array([5])
        fig, ax = plot_histogram(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_bar_single_category(self):
        """Test that plot_bar handles single category."""
        categories = ["A"]
        values = [10]
        fig, ax = plot_bar(categories, values)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestLargeData:
    """Tests for large data handling."""

    def test_plot_timeseries_large_data(self):
        """Test that plot_timeseries handles large datasets."""
        n = 10000
        data = pd.Series(np.random.randn(n), index=pd.date_range("2020-01-01", periods=n))
        fig, ax = plot_timeseries(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_histogram_large_data(self):
        """Test that plot_histogram handles large datasets."""
        data = np.random.randn(10000)
        fig, ax = plot_histogram(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

