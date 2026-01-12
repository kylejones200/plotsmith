"""Performance benchmarks for PlotSmith."""

# Use Agg backend for testing (non-interactive)
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from plotsmith.workflows.workflows import (
    plot_bar,
    plot_heatmap,
    plot_histogram,
    plot_scatter,
    plot_timeseries,
)

pytestmark = pytest.mark.slow


class TestPerformance:
    """Performance benchmarks for common operations."""

    def test_plot_timeseries_large_dataset(self, benchmark):
        """Benchmark time series plotting with large dataset."""
        n = 10000
        data = pd.Series(
            np.random.randn(n), index=pd.date_range("2020-01-01", periods=n, freq="H")
        )

        def run():
            fig, ax = plot_timeseries(data)
            plt.close(fig)
            return fig, ax

        result = benchmark(run)
        assert result[0] is not None

    def test_plot_timeseries_multiple_series(self, benchmark):
        """Benchmark time series plotting with multiple series."""
        n = 1000
        df = pd.DataFrame(
            {
                "A": np.random.randn(n),
                "B": np.random.randn(n),
                "C": np.random.randn(n),
                "D": np.random.randn(n),
            },
            index=pd.date_range("2020-01-01", periods=n, freq="H"),
        )

        def run():
            fig, ax = plot_timeseries(df)
            plt.close(fig)
            return fig, ax

        result = benchmark(run)
        assert result[0] is not None

    def test_plot_histogram_large_dataset(self, benchmark):
        """Benchmark histogram plotting with large dataset."""
        data = np.random.randn(50000)

        def run():
            fig, ax = plot_histogram(data)
            plt.close(fig)
            return fig, ax

        result = benchmark(run)
        assert result[0] is not None

    def test_plot_scatter_large_dataset(self, benchmark):
        """Benchmark scatter plotting with large dataset."""
        n = 10000
        df = pd.DataFrame(
            {
                "x": np.random.randn(n),
                "y": np.random.randn(n),
            }
        )

        def run():
            fig, ax = plot_scatter(df, x="x", y="y")
            plt.close(fig)
            return fig, ax

        result = benchmark(run)
        assert result[0] is not None

    def test_plot_heatmap_large_matrix(self, benchmark):
        """Benchmark heatmap plotting with large matrix."""
        data = np.random.rand(100, 100)

        def run():
            fig, ax = plot_heatmap(data)
            plt.close(fig)
            return fig, ax

        result = benchmark(run)
        assert result[0] is not None

    def test_plot_bar_many_categories(self, benchmark):
        """Benchmark bar plotting with many categories."""
        n = 1000
        categories = [f"Cat_{i}" for i in range(n)]
        values = np.random.randn(n)

        def run():
            fig, ax = plot_bar(categories, values)
            plt.close(fig)
            return fig, ax

        result = benchmark(run)
        assert result[0] is not None

    def test_plot_timeseries_with_bands(self, benchmark):
        """Benchmark time series plotting with confidence bands."""
        n = 5000
        data = pd.Series(
            np.random.randn(n), index=pd.date_range("2020-01-01", periods=n, freq="H")
        )
        lower = data - 0.5
        upper = data + 0.5
        bands = {"95% CI": (lower, upper)}

        def run():
            fig, ax = plot_timeseries(data, bands=bands)
            plt.close(fig)
            return fig, ax

        result = benchmark(run)
        assert result[0] is not None

