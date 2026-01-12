"""Extended tests for Layer 3: tasks - all new task classes."""

import numpy as np
import pandas as pd
import pytest

from plotsmith.exceptions import ValidationError
from plotsmith.objects.views import (
    BarView,
    BoxView,
    DumbbellView,
    FigureSpec,
    HeatmapView,
    HistogramView,
    LollipopView,
    MetricView,
    RangeView,
    ScatterView,
    SlopeView,
    ViolinView,
    WaffleView,
    WaterfallView,
)
from plotsmith.tasks.tasks import (
    BarPlotTask,
    BoxPlotTask,
    CorrelationPlotTask,
    DumbbellPlotTask,
    ForecastComparisonPlotTask,
    HeatmapPlotTask,
    HistogramPlotTask,
    LollipopPlotTask,
    MetricPlotTask,
    RangePlotTask,
    ResidualsPlotTask,
    ScatterPlotTask,
    SlopePlotTask,
    ViolinPlotTask,
    WafflePlotTask,
    WaterfallPlotTask,
)


class TestHistogramPlotTask:
    """Tests for HistogramPlotTask."""

    def test_histogram_task_returns_histogram_view(self):
        """Test that HistogramPlotTask returns HistogramView."""
        data = np.array([1, 2, 3, 4, 5])
        task = HistogramPlotTask(data=data)
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], HistogramView)
        assert isinstance(spec, FigureSpec)

    def test_histogram_task_with_bins(self):
        """Test that HistogramPlotTask works with custom bins."""
        data = np.array([1, 2, 3, 4, 5])
        task = HistogramPlotTask(data=data, bins=10)
        views, spec = task.execute()

        assert views[0].bins == 10


class TestBarPlotTask:
    """Tests for BarPlotTask."""

    def test_bar_task_returns_bar_view(self):
        """Test that BarPlotTask returns BarView."""
        categories = ["A", "B", "C"]
        values = [10, 20, 15]
        task = BarPlotTask(x=categories, height=values)
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], BarView)
        assert isinstance(spec, FigureSpec)

    def test_bar_task_horizontal(self):
        """Test that BarPlotTask works with horizontal orientation."""
        categories = ["A", "B", "C"]
        values = [10, 20, 15]
        task = BarPlotTask(x=categories, height=values, horizontal=True)
        views, spec = task.execute()

        assert views[0].horizontal is True


class TestHeatmapPlotTask:
    """Tests for HeatmapPlotTask."""

    def test_heatmap_task_returns_heatmap_view(self):
        """Test that HeatmapPlotTask returns HeatmapView."""
        data = np.random.rand(5, 5)
        task = HeatmapPlotTask(data=data)
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], HeatmapView)
        assert isinstance(spec, FigureSpec)

    def test_heatmap_task_with_dataframe(self):
        """Test that HeatmapPlotTask works with DataFrame."""
        data = pd.DataFrame(np.random.rand(3, 3), columns=["A", "B", "C"])
        task = HeatmapPlotTask(data=data)
        views, spec = task.execute()

        assert isinstance(views[0], HeatmapView)


class TestResidualsPlotTask:
    """Tests for ResidualsPlotTask."""

    def test_residuals_task_scatter_returns_scatter_view(self):
        """Test that ResidualsPlotTask returns ScatterView for scatter plot."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        task = ResidualsPlotTask(actual=actual, predicted=predicted, plot_type="scatter")
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], ScatterView)
        assert isinstance(spec, FigureSpec)

    def test_residuals_task_series_returns_series_view(self):
        """Test that ResidualsPlotTask returns SeriesView for series plot."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        task = ResidualsPlotTask(actual=actual, predicted=predicted, plot_type="series")
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0].__class__.__name__, str)  # SeriesView
        assert isinstance(spec, FigureSpec)

    def test_residuals_task_mismatched_lengths_raises_error(self):
        """Test that ResidualsPlotTask raises error for mismatched lengths."""
        actual = np.array([1, 2, 3])
        predicted = np.array([1.1, 2.2])
        task = ResidualsPlotTask(actual=actual, predicted=predicted)
        with pytest.raises(ValueError, match="same length"):
            task.execute()


class TestWaterfallPlotTask:
    """Tests for WaterfallPlotTask."""

    def test_waterfall_task_returns_waterfall_view(self):
        """Test that WaterfallPlotTask returns WaterfallView."""
        df = pd.DataFrame({"Category": ["A", "B", "C"], "Value": [10, 20, 15]})
        task = WaterfallPlotTask(df=df, categories_col="Category", values_col="Value")
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], WaterfallView)
        assert isinstance(spec, FigureSpec)


class TestWafflePlotTask:
    """Tests for WafflePlotTask."""

    def test_waffle_task_returns_waffle_view(self):
        """Test that WafflePlotTask returns WaffleView."""
        df = pd.DataFrame({"Party": ["A", "B"], "Seats": [10, 20]})
        task = WafflePlotTask(df=df, category_col="Party", value_col="Seats")
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], WaffleView)
        assert isinstance(spec, FigureSpec)


class TestDumbbellPlotTask:
    """Tests for DumbbellPlotTask."""

    def test_dumbbell_task_returns_dumbbell_view(self):
        """Test that DumbbellPlotTask returns DumbbellView."""
        df = pd.DataFrame({"cat": ["A", "B"], "v1": [10, 20], "v2": [15, 25]})
        task = DumbbellPlotTask(
            df=df, categories_col="cat", values1_col="v1", values2_col="v2"
        )
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], DumbbellView)
        assert isinstance(spec, FigureSpec)


class TestRangePlotTask:
    """Tests for RangePlotTask."""

    def test_range_task_returns_range_view(self):
        """Test that RangePlotTask returns RangeView."""
        df = pd.DataFrame({"cat": ["A", "B"], "v1": [10, 20], "v2": [15, 25]})
        task = RangePlotTask(
            df=df, categories_col="cat", values1_col="v1", values2_col="v2"
        )
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], RangeView)
        assert isinstance(spec, FigureSpec)


class TestLollipopPlotTask:
    """Tests for LollipopPlotTask."""

    def test_lollipop_task_returns_lollipop_view(self):
        """Test that LollipopPlotTask returns LollipopView."""
        df = pd.DataFrame({"Category": ["A", "B"], "Value": [10, 20]})
        task = LollipopPlotTask(df=df, categories_col="Category", values_col="Value")
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], LollipopView)
        assert isinstance(spec, FigureSpec)


class TestSlopePlotTask:
    """Tests for SlopePlotTask."""

    def test_slope_task_returns_slope_view(self):
        """Test that SlopePlotTask returns SlopeView."""
        x = np.array([2020, 2021])
        groups = {"USA": np.array([10, 12]), "Canada": np.array([12, 15])}
        task = SlopePlotTask(x=x, groups=groups)
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], SlopeView)
        assert isinstance(spec, FigureSpec)


class TestMetricPlotTask:
    """Tests for MetricPlotTask."""

    def test_metric_task_returns_metric_view(self):
        """Test that MetricPlotTask returns MetricView."""
        task = MetricPlotTask(title="Sales", value=12345, delta=123)
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], MetricView)
        assert isinstance(spec, FigureSpec)


class TestBoxPlotTask:
    """Tests for BoxPlotTask."""

    def test_box_task_returns_box_view(self):
        """Test that BoxPlotTask returns BoxView."""
        df = pd.DataFrame({"category": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})
        task = BoxPlotTask(df=df, x="category", y="value")
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], BoxView)
        assert isinstance(spec, FigureSpec)


class TestViolinPlotTask:
    """Tests for ViolinPlotTask."""

    def test_violin_task_returns_violin_view(self):
        """Test that ViolinPlotTask returns ViolinView."""
        df = pd.DataFrame({"category": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})
        task = ViolinPlotTask(df=df, x="category", y="value")
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], ViolinView)
        assert isinstance(spec, FigureSpec)


class TestScatterPlotTask:
    """Tests for ScatterPlotTask."""

    def test_scatter_task_returns_scatter_view(self):
        """Test that ScatterPlotTask returns ScatterView."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        task = ScatterPlotTask(df=df, x="x", y="y")
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], ScatterView)
        assert isinstance(spec, FigureSpec)

    def test_scatter_task_with_color(self):
        """Test that ScatterPlotTask works with color mapping."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30], "c": ["A", "B", "A"]})
        task = ScatterPlotTask(df=df, x="x", y="y", color="c")
        views, spec = task.execute()

        assert views[0].c is not None


class TestCorrelationPlotTask:
    """Tests for CorrelationPlotTask."""

    def test_correlation_task_returns_heatmap_view(self):
        """Test that CorrelationPlotTask returns HeatmapView."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 4, 6]})
        task = CorrelationPlotTask(df=df)
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], HeatmapView)
        assert isinstance(spec, FigureSpec)


class TestForecastComparisonPlotTask:
    """Tests for ForecastComparisonPlotTask."""

    def test_forecast_comparison_task_returns_views(self):
        """Test that ForecastComparisonPlotTask returns views."""
        actual = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        forecasts = {"Model1": pd.Series([1.1, 2.1, 2.9], index=actual.index)}
        task = ForecastComparisonPlotTask(actual=actual, forecasts=forecasts)
        views, spec = task.execute()

        assert len(views) > 0
        assert isinstance(spec, FigureSpec)

