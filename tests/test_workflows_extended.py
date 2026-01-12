"""Extended tests for Layer 4: workflows - all new chart types."""

import numpy as np
import pandas as pd
import pytest

# Use Agg backend for testing (non-interactive)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotsmith.workflows.workflows import (
    figure,
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
    plot_violin,
    plot_waffle,
    plot_waterfall,
    small_multiples,
)


class TestPlotHistogram:
    """Tests for plot_histogram workflow."""

    def test_plot_histogram_returns_fig_ax(self):
        """Test that plot_histogram returns fig and ax."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        fig, ax = plot_histogram(data)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_histogram_with_series(self):
        """Test that plot_histogram works with pandas Series."""
        data = pd.Series([1, 2, 3, 4, 5])
        fig, ax = plot_histogram(data)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_histogram_with_multiple(self):
        """Test that plot_histogram works with multiple datasets."""
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([6, 7, 8, 9, 10])
        fig, ax = plot_histogram([data1, data2], labels=["A", "B"])

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_histogram_saves_file(self, tmp_path):
        """Test that plot_histogram can save to file."""
        data = np.array([1, 2, 3, 4, 5])
        save_path = tmp_path / "test.png"
        fig, ax = plot_histogram(data, save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotBar:
    """Tests for plot_bar workflow."""

    def test_plot_bar_returns_fig_ax(self):
        """Test that plot_bar returns fig and ax."""
        categories = ["A", "B", "C"]
        values = [10, 20, 15]
        fig, ax = plot_bar(categories, values)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_bar_horizontal(self):
        """Test that plot_bar works with horizontal orientation."""
        categories = ["A", "B", "C"]
        values = [10, 20, 15]
        fig, ax = plot_bar(categories, values, horizontal=True)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_bar_saves_file(self, tmp_path):
        """Test that plot_bar can save to file."""
        categories = ["A", "B", "C"]
        values = [10, 20, 15]
        save_path = tmp_path / "test.png"
        fig, ax = plot_bar(categories, values, save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotHeatmap:
    """Tests for plot_heatmap workflow."""

    def test_plot_heatmap_returns_fig_ax(self):
        """Test that plot_heatmap returns fig and ax."""
        data = np.random.rand(5, 5)
        fig, ax = plot_heatmap(data)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_heatmap_with_dataframe(self):
        """Test that plot_heatmap works with DataFrame."""
        data = pd.DataFrame(np.random.rand(3, 3), columns=["A", "B", "C"], index=["X", "Y", "Z"])
        fig, ax = plot_heatmap(data)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_heatmap_with_annotations(self):
        """Test that plot_heatmap works with annotations."""
        data = np.array([[1, 2], [3, 4]])
        fig, ax = plot_heatmap(data, annotate=True)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_heatmap_saves_file(self, tmp_path):
        """Test that plot_heatmap can save to file."""
        data = np.random.rand(5, 5)
        save_path = tmp_path / "test.png"
        fig, ax = plot_heatmap(data, save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotResiduals:
    """Tests for plot_residuals workflow."""

    def test_plot_residuals_scatter_returns_fig_ax(self):
        """Test that plot_residuals returns fig and ax for scatter plot."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        fig, ax = plot_residuals(actual, predicted, plot_type="scatter")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_residuals_series_returns_fig_ax(self):
        """Test that plot_residuals returns fig and ax for series plot."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        x = np.arange(len(actual))
        fig, ax = plot_residuals(actual, predicted, x=x, plot_type="series")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_residuals_saves_file(self, tmp_path):
        """Test that plot_residuals can save to file."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        save_path = tmp_path / "test.png"
        fig, ax = plot_residuals(actual, predicted, save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotModelComparison:
    """Tests for plot_model_comparison workflow."""

    def test_plot_model_comparison_returns_fig_ax(self):
        """Test that plot_model_comparison returns fig and ax."""
        data = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2020-01-01", periods=5))
        models = {
            "Model1": np.array([1.1, 2.1, 2.9, 4.1, 4.9]),
            "Model2": np.array([0.9, 2.2, 3.1, 3.9, 5.1]),
        }
        fig, ax = plot_model_comparison(data, models)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_model_comparison_saves_file(self, tmp_path):
        """Test that plot_model_comparison can save to file."""
        data = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2020-01-01", periods=5))
        models = {"Model1": np.array([1.1, 2.1, 2.9, 4.1, 4.9])}
        save_path = tmp_path / "test.png"
        fig, ax = plot_model_comparison(data, models, save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotWaterfall:
    """Tests for plot_waterfall workflow."""

    def test_plot_waterfall_returns_fig_ax(self):
        """Test that plot_waterfall returns fig and ax."""
        df = pd.DataFrame(
            {
                "Category": ["Revenue", "COGS", "Operating Expenses", "Taxes", "Net Profit"],
                "Value": [100000, -50000, -20000, -10000, 0],
            }
        )
        fig, ax = plot_waterfall(df, categories_col="Category", values_col="Value")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_waterfall_with_measures(self):
        """Test that plot_waterfall works with measure types."""
        df = pd.DataFrame(
            {
                "Category": ["Year beginning", "Profit1", "Loss1", "Q1", "Profit2", "Loss2", "Q2"],
                "Value": [100, 50, -20, 0, 40, -10, 0],
                "measure": ["absolute", "relative", "relative", "total", "relative", "relative", "total"],
            }
        )
        fig, ax = plot_waterfall(df, categories_col="Category", values_col="Value", measure_col="measure")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_waterfall_saves_file(self, tmp_path):
        """Test that plot_waterfall can save to file."""
        df = pd.DataFrame({"Category": ["A", "B", "C"], "Value": [10, 20, 15]})
        save_path = tmp_path / "test.png"
        fig, ax = plot_waterfall(df, categories_col="Category", values_col="Value", save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotWaffle:
    """Tests for plot_waffle workflow."""

    def test_plot_waffle_returns_fig_ax(self):
        """Test that plot_waffle returns fig and ax."""
        df = pd.DataFrame({"Party": ["A", "B", "C"], "Seats": [10, 20, 15]})
        fig, ax = plot_waffle(df, category_col="Party", value_col="Seats")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_waffle_with_custom_grid(self):
        """Test that plot_waffle works with custom grid size."""
        df = pd.DataFrame({"Party": ["A", "B"], "Seats": [10, 20]})
        fig, ax = plot_waffle(df, category_col="Party", value_col="Seats", rows=5, columns=6)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_waffle_saves_file(self, tmp_path):
        """Test that plot_waffle can save to file."""
        df = pd.DataFrame({"Party": ["A", "B"], "Seats": [10, 20]})
        save_path = tmp_path / "test.png"
        fig, ax = plot_waffle(df, category_col="Party", value_col="Seats", save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotDumbbell:
    """Tests for plot_dumbbell workflow."""

    def test_plot_dumbbell_returns_fig_ax(self):
        """Test that plot_dumbbell returns fig and ax."""
        df = pd.DataFrame(
            {
                "country": ["France", "Germany", "Greece"],
                "1952": [67.41, 67.5, 65.86],
                "2002": [79.59, 78.67, 78.25],
            }
        )
        fig, ax = plot_dumbbell(df, categories_col="country", values1_col="1952", values2_col="2002")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_dumbbell_vertical(self):
        """Test that plot_dumbbell works with vertical orientation."""
        df = pd.DataFrame({"cat": ["A", "B"], "v1": [10, 20], "v2": [15, 25]})
        fig, ax = plot_dumbbell(df, categories_col="cat", values1_col="v1", values2_col="v2", orientation="v")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_dumbbell_saves_file(self, tmp_path):
        """Test that plot_dumbbell can save to file."""
        df = pd.DataFrame({"cat": ["A", "B"], "v1": [10, 20], "v2": [15, 25]})
        save_path = tmp_path / "test.png"
        fig, ax = plot_dumbbell(df, categories_col="cat", values1_col="v1", values2_col="v2", save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotRange:
    """Tests for plot_range workflow."""

    def test_plot_range_returns_fig_ax(self):
        """Test that plot_range returns fig and ax."""
        df = pd.DataFrame({"cat": ["A", "B"], "v1": [10, 20], "v2": [15, 25]})
        fig, ax = plot_range(df, categories_col="cat", values1_col="v1", values2_col="v2")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_range_saves_file(self, tmp_path):
        """Test that plot_range can save to file."""
        df = pd.DataFrame({"cat": ["A", "B"], "v1": [10, 20], "v2": [15, 25]})
        save_path = tmp_path / "test.png"
        fig, ax = plot_range(df, categories_col="cat", values1_col="v1", values2_col="v2", save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotLollipop:
    """Tests for plot_lollipop workflow."""

    def test_plot_lollipop_returns_fig_ax(self):
        """Test that plot_lollipop returns fig and ax."""
        df = pd.DataFrame({"Category": ["A", "B", "C"], "Value": [10, 25, 15]})
        fig, ax = plot_lollipop(df, categories_col="Category", values_col="Value")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_lollipop_horizontal(self):
        """Test that plot_lollipop works with horizontal orientation."""
        df = pd.DataFrame({"Category": ["A", "B"], "Value": [10, 20]})
        fig, ax = plot_lollipop(df, categories_col="Category", values_col="Value", horizontal=True)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_lollipop_saves_file(self, tmp_path):
        """Test that plot_lollipop can save to file."""
        df = pd.DataFrame({"Category": ["A", "B"], "Value": [10, 20]})
        save_path = tmp_path / "test.png"
        fig, ax = plot_lollipop(df, categories_col="Category", values_col="Value", save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotSlope:
    """Tests for plot_slope workflow."""

    def test_plot_slope_long_format_returns_fig_ax(self):
        """Test that plot_slope returns fig and ax with long format."""
        df = pd.DataFrame(
            {
                "year": [2020, 2020, 2020, 2023, 2023, 2023],
                "country": ["USA", "Canada", "Mexico", "USA", "Canada", "Mexico"],
                "gdp": [20.94, 1.64, 1.07, 22.99, 1.99, 1.27],
            }
        )
        fig, ax = plot_slope(data_frame=df, x="year", y="gdp", group="country")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_slope_wide_format_returns_fig_ax(self):
        """Test that plot_slope returns fig and ax with wide format."""
        df = pd.DataFrame({"Year": [2020, 2021], "USA": [10, 12], "Canada": [12, 15]})
        fig, ax = plot_slope(df=df, categories_col="Year", values_cols=["USA", "Canada"])

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_slope_saves_file(self, tmp_path):
        """Test that plot_slope can save to file."""
        df = pd.DataFrame({"Year": [2020, 2021], "USA": [10, 12], "Canada": [12, 15]})
        save_path = tmp_path / "test.png"
        fig, ax = plot_slope(df=df, categories_col="Year", values_cols=["USA", "Canada"], save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotMetric:
    """Tests for plot_metric workflow."""

    def test_plot_metric_returns_fig_ax(self):
        """Test that plot_metric returns fig and ax."""
        fig, ax = plot_metric(title="Sales", value=12345, delta=123, prefix="$")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_metric_with_negative_delta(self):
        """Test that plot_metric works with negative delta."""
        fig, ax = plot_metric(title="Sales", value=12345, delta=-123, prefix="$")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_metric_saves_file(self, tmp_path):
        """Test that plot_metric can save to file."""
        save_path = tmp_path / "test.png"
        fig, ax = plot_metric(title="Sales", value=12345, save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotBox:
    """Tests for plot_box workflow."""

    def test_plot_box_returns_fig_ax(self):
        """Test that plot_box returns fig and ax."""
        df = pd.DataFrame({"category": ["A", "A", "B", "B", "C", "C"], "value": [1, 2, 3, 4, 5, 6]})
        fig, ax = plot_box(df, x="category", y="value")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_box_with_means(self):
        """Test that plot_box works with show_means."""
        df = pd.DataFrame({"cat": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
        fig, ax = plot_box(df, x="cat", y="val", show_means=True)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_box_saves_file(self, tmp_path):
        """Test that plot_box can save to file."""
        df = pd.DataFrame({"cat": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
        save_path = tmp_path / "test.png"
        fig, ax = plot_box(df, x="cat", y="val", save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotViolin:
    """Tests for plot_violin workflow."""

    def test_plot_violin_returns_fig_ax(self):
        """Test that plot_violin returns fig and ax."""
        df = pd.DataFrame({"category": ["A", "A", "B", "B", "C", "C"], "value": [1, 2, 3, 4, 5, 6]})
        fig, ax = plot_violin(df, x="category", y="value")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_violin_with_means(self):
        """Test that plot_violin works with show_means."""
        df = pd.DataFrame({"cat": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
        fig, ax = plot_violin(df, x="cat", y="val", show_means=True)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_violin_saves_file(self, tmp_path):
        """Test that plot_violin can save to file."""
        df = pd.DataFrame({"cat": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
        save_path = tmp_path / "test.png"
        fig, ax = plot_violin(df, x="cat", y="val", save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotScatter:
    """Tests for plot_scatter workflow."""

    def test_plot_scatter_returns_fig_ax(self):
        """Test that plot_scatter returns fig and ax."""
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
        fig, ax = plot_scatter(df, x="x", y="y")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_scatter_with_color(self):
        """Test that plot_scatter works with color mapping."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30], "category": ["A", "B", "A"]})
        fig, ax = plot_scatter(df, x="x", y="y", color="category")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_scatter_with_size(self):
        """Test that plot_scatter works with size mapping."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30], "size": [10, 20, 30]})
        fig, ax = plot_scatter(df, x="x", y="y", size="size")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_scatter_saves_file(self, tmp_path):
        """Test that plot_scatter can save to file."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        save_path = tmp_path / "test.png"
        fig, ax = plot_scatter(df, x="x", y="y", save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotCorrelation:
    """Tests for plot_correlation workflow."""

    def test_plot_correlation_returns_fig_ax(self):
        """Test that plot_correlation returns fig and ax."""
        df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 4, 6, 8], "C": [1, 3, 5, 7]})
        fig, ax = plot_correlation(df)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_correlation_with_method(self):
        """Test that plot_correlation works with different methods."""
        df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 4, 6, 8]})
        fig, ax = plot_correlation(df, method="spearman")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_correlation_saves_file(self, tmp_path):
        """Test that plot_correlation can save to file."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 4, 6]})
        save_path = tmp_path / "test.png"
        fig, ax = plot_correlation(df, save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestPlotForecastComparison:
    """Tests for plot_forecast_comparison workflow."""

    def test_plot_forecast_comparison_returns_fig_ax(self):
        """Test that plot_forecast_comparison returns fig and ax."""
        actual = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2020-01-01", periods=5))
        forecasts = {
            "Model1": pd.Series([1.1, 2.1, 2.9, 4.1, 4.9], index=actual.index),
            "Model2": pd.Series([0.9, 2.2, 3.1, 3.9, 5.1], index=actual.index),
        }
        fig, ax = plot_forecast_comparison(actual, forecasts)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_forecast_comparison_with_intervals(self):
        """Test that plot_forecast_comparison works with intervals."""
        actual = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        forecasts = {"Model1": pd.Series([1.1, 2.1, 2.9], index=actual.index)}
        intervals = {
            "Model1": (
                pd.Series([0.9, 1.9, 2.7], index=actual.index),
                pd.Series([1.3, 2.3, 3.1], index=actual.index),
            )
        }
        fig, ax = plot_forecast_comparison(actual, forecasts, intervals=intervals)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_forecast_comparison_saves_file(self, tmp_path):
        """Test that plot_forecast_comparison can save to file."""
        actual = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        forecasts = {"Model1": pd.Series([1.1, 2.1, 2.9], index=actual.index)}
        save_path = tmp_path / "test.png"
        fig, ax = plot_forecast_comparison(actual, forecasts, save_path=save_path)

        assert save_path.exists()
        plt.close(fig)


class TestFigure:
    """Tests for figure workflow."""

    def test_figure_returns_fig_ax(self):
        """Test that figure returns fig and ax."""
        fig, ax = figure()

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_figure_with_subplots(self):
        """Test that figure works with multiple subplots."""
        fig, axes = figure(nrows=2, ncols=2)

        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert len(axes.flatten()) == 4
        plt.close(fig)


class TestSmallMultiples:
    """Tests for small_multiples workflow."""

    def test_small_multiples_returns_fig_axes(self):
        """Test that small_multiples returns fig and list of axes."""
        fig, axes = small_multiples(6)

        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, list)
        assert len(axes) == 6
        plt.close(fig)

    def test_small_multiples_with_custom_cols(self):
        """Test that small_multiples works with custom columns."""
        fig, axes = small_multiples(5, ncols=2)

        assert isinstance(fig, plt.Figure)
        assert len(axes) == 5
        plt.close(fig)

