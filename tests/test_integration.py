"""Integration tests for complete PlotSmith workflows."""

# Use Agg backend for testing (non-interactive)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotsmith.workflows.workflows import (
    plot_backtest,
    plot_box,
    plot_correlation,
    plot_forecast_comparison,
    plot_heatmap,
    plot_histogram,
    plot_model_comparison,
    plot_residuals,
    plot_scatter,
    plot_timeseries,
    plot_violin,
)


class TestCompleteWorkflows:
    """Integration tests for complete plotting workflows."""

    def test_timeseries_with_bands_workflow(self):
        """Test complete time series workflow with confidence bands."""
        # Create data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = pd.Series(np.random.randn(100).cumsum(), index=dates)
        lower = data - 1.96 * np.random.rand(100)
        upper = data + 1.96 * np.random.rand(100)

        # Plot
        fig, ax = plot_timeseries(
            data,
            bands={
                "95% CI": (pd.Series(lower, index=dates), pd.Series(upper, index=dates))
            },
            title="Time Series with Confidence Bands",
            save_path=None,
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_model_evaluation_workflow(self):
        """Test complete model evaluation workflow."""
        # Create backtest results
        results = pd.DataFrame(
            {
                "y_true": np.random.randn(100),
                "y_pred": np.random.randn(100) * 0.8,
                "fold_id": np.repeat([0, 1, 2], [33, 33, 34]),
            }
        )

        # Plot backtest
        fig1, ax1 = plot_backtest(results, fold_id_col="fold_id")
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        # Plot residuals
        fig2, ax2 = plot_residuals(
            results["y_true"].values,
            results["y_pred"].values,
            plot_type="scatter",
        )
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

    def test_statistical_analysis_workflow(self):
        """Test complete statistical analysis workflow."""
        # Create data
        df = pd.DataFrame(
            {
                "category": np.repeat(["A", "B", "C"], 50),
                "value": np.concatenate(
                    [
                        np.random.randn(50) + 0,
                        np.random.randn(50) + 2,
                        np.random.randn(50) + 4,
                    ]
                ),
            }
        )

        # Box plot
        fig1, ax1 = plot_box(df, x="category", y="value")
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        # Violin plot
        fig2, ax2 = plot_violin(df, x="category", y="value")
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

        # Histogram
        fig3, ax3 = plot_histogram(df["value"].values)
        assert isinstance(fig3, plt.Figure)
        plt.close(fig3)

    def test_correlation_analysis_workflow(self):
        """Test complete correlation analysis workflow."""
        # Create correlated data
        np.random.seed(42)
        n = 100
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(n),
                "feature2": np.random.randn(n),
                "feature3": np.random.randn(n),
                "target": np.random.randn(n),
            }
        )

        # Correlation heatmap
        fig, ax = plot_correlation(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_forecast_comparison_workflow(self):
        """Test complete forecast comparison workflow."""
        # Create time series data
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        actual = pd.Series(np.random.randn(50).cumsum(), index=dates)

        # Create forecasts
        forecasts = {
            "Model1": pd.Series(actual.values + np.random.randn(50) * 0.1, index=dates),
            "Model2": pd.Series(actual.values + np.random.randn(50) * 0.2, index=dates),
        }

        # Plot comparison
        fig, ax = plot_forecast_comparison(actual, forecasts)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_scatter_analysis_workflow(self):
        """Test complete scatter analysis workflow."""
        # Create data with categories and sizes
        df = pd.DataFrame(
            {
                "x": np.random.randn(100),
                "y": np.random.randn(100),
                "category": np.repeat(["A", "B"], 50),
                "size": np.random.rand(100) * 100,
            }
        )

        # Scatter with color
        fig1, ax1 = plot_scatter(df, x="x", y="y", color="category")
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        # Scatter with size
        fig2, ax2 = plot_scatter(df, x="x", y="y", size="size")
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

    def test_heatmap_workflow(self):
        """Test complete heatmap workflow."""
        # Create correlation matrix
        data = pd.DataFrame(np.random.randn(10, 5))
        corr = data.corr()

        # Plot heatmap
        fig, ax = plot_heatmap(corr, annotate=True, cmap="RdYlGn")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_model_comparison_workflow(self):
        """Test complete model comparison workflow."""
        # Create time series
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        actual = pd.Series(np.random.randn(100).cumsum(), index=dates)

        # Create model predictions
        models = {
            "LSTM": pd.Series(actual.values + np.random.randn(100) * 0.1, index=dates),
            "XGBoost": pd.Series(
                actual.values + np.random.randn(100) * 0.15, index=dates
            ),
            "Prophet": pd.Series(
                actual.values + np.random.randn(100) * 0.2, index=dates
            ),
        }

        # Plot comparison
        fig, ax = plot_model_comparison(actual, models, test_start_idx=80)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
