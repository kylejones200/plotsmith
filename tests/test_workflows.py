"""Tests for Layer 4: workflows."""

# Use Agg backend for testing (non-interactive)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from plotsmith.workflows.workflows import (
    figure,
    plot_backtest,
    plot_bar,
    plot_box,
    plot_correlation,
    plot_dumbbell,
    plot_forecast_comparison,
    plot_heatmap,
    plot_histogram,
    plot_lollipop,
    plot_metric,
    plot_model_comparison,
    plot_range,
    plot_residuals,
    plot_scatter,
    plot_slope,
    plot_timeseries,
    plot_violin,
    plot_waffle,
    plot_waterfall,
)


class TestPlotTimeseries:
    """Tests for plot_timeseries workflow."""

    def test_plot_timeseries_returns_fig_ax(self):
        """Test that plot_timeseries returns fig and ax."""
        data = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2020-01-01", periods=5))
        fig, ax = plot_timeseries(data)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_timeseries_with_dataframe(self):
        """Test that plot_timeseries works with DataFrame."""
        data = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]}, index=pd.date_range("2020-01-01", periods=3)
        )
        fig, ax = plot_timeseries(data)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_timeseries_with_bands(self):
        """Test that plot_timeseries works with confidence bands."""
        data = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        lower = pd.Series([0.5, 1.5, 2.5], index=data.index)
        upper = pd.Series([1.5, 2.5, 3.5], index=data.index)
        bands = {"confidence": (lower, upper)}

        fig, ax = plot_timeseries(data, bands=bands)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_timeseries_saves_file(self, tmp_path):
        """Test that plot_timeseries can save to file."""
        data = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        save_path = tmp_path / "test.png"
        fig, ax = plot_timeseries(data, save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotBacktest:
    """Tests for plot_backtest workflow."""

    def test_plot_backtest_returns_fig_ax(self):
        """Test that plot_backtest returns fig and ax."""
        results = pd.DataFrame({"y_true": [1, 2, 3], "y_pred": [1.1, 2.1, 2.9]})
        fig, ax = plot_backtest(results)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_backtest_with_fold_id(self):
        """Test that plot_backtest works with fold_id."""
        results = pd.DataFrame(
            {
                "y_true": [1, 2, 3, 4, 5],
                "y_pred": [1.1, 2.1, 2.9, 4.1, 4.9],
                "fold_id": [0, 0, 1, 1, 2],
            }
        )
        fig, ax = plot_backtest(results, fold_id_col="fold_id")
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_backtest_saves_file(self, tmp_path):
        """Test that plot_backtest can save to file."""
        results = pd.DataFrame({"y_true": [1, 2, 3], "y_pred": [1.1, 2.1, 2.9]})
        save_path = tmp_path / "test.png"
        fig, ax = plot_backtest(results, save_path=save_path)

        assert save_path.exists()
        plt.close(fig)

