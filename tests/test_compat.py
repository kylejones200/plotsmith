"""Tests for compatibility layer."""

import warnings

import pandas as pd
import pytest

# Use Agg backend for testing (non-interactive)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotsmith.compat.signalplot import plot_prediction_scatter, plot_series


class TestCompatWrappers:
    """Tests for compatibility wrappers."""

    def test_plot_series_emits_deprecation_warning(self):
        """Test that plot_series emits DeprecationWarning."""
        data = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        with pytest.warns(DeprecationWarning, match="plot_series is deprecated"):
            fig, ax = plot_series(data)
            plt.close(fig)

    def test_plot_series_still_works(self):
        """Test that plot_series still functions correctly."""
        data = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fig, ax = plot_series(data)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_prediction_scatter_emits_deprecation_warning(self):
        """Test that plot_prediction_scatter emits DeprecationWarning."""
        results = pd.DataFrame({"y_true": [1, 2, 3], "y_pred": [1.1, 2.1, 2.9]})
        with pytest.warns(DeprecationWarning, match="plot_prediction_scatter is deprecated"):
            fig, ax = plot_prediction_scatter(results)
            plt.close(fig)

    def test_plot_prediction_scatter_still_works(self):
        """Test that plot_prediction_scatter still functions correctly."""
        results = pd.DataFrame({"y_true": [1, 2, 3], "y_pred": [1.1, 2.1, 2.9]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fig, ax = plot_prediction_scatter(results)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

