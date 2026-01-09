"""Tests for Layer 3: tasks."""

import numpy as np
import pandas as pd
import pytest

from plotsmith.objects.views import BandView, FigureSpec, ScatterView, SeriesView
from plotsmith.tasks.tasks import BacktestPlotTask, TimeseriesPlotTask


class TestTimeseriesPlotTask:
    """Tests for TimeseriesPlotTask."""

    def test_series_input_returns_series_view(self):
        """Test that Series input returns SeriesView."""
        data = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2020-01-01", periods=5))
        task = TimeseriesPlotTask(data=data)
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], SeriesView)
        assert isinstance(spec, FigureSpec)

    def test_dataframe_input_returns_multiple_series_views(self):
        """Test that DataFrame input returns multiple SeriesView objects."""
        data = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]}, index=pd.date_range("2020-01-01", periods=3)
        )
        task = TimeseriesPlotTask(data=data)
        views, spec = task.execute()

        assert len(views) == 2
        assert all(isinstance(v, SeriesView) for v in views)
        assert views[0].label == "A"
        assert views[1].label == "B"

    def test_with_bands_returns_band_views(self):
        """Test that bands are included in the views."""
        data = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        lower = pd.Series([0.5, 1.5, 2.5], index=data.index)
        upper = pd.Series([1.5, 2.5, 3.5], index=data.index)
        bands = {"confidence": (lower, upper)}

        task = TimeseriesPlotTask(data=data, bands=bands)
        views, spec = task.execute()

        assert len(views) == 2  # One series, one band
        assert isinstance(views[0], SeriesView)
        assert isinstance(views[1], BandView)

    def test_invalid_data_type_raises_error(self):
        """Test that invalid data type raises TypeError."""
        task = TimeseriesPlotTask(data=[1, 2, 3])  # Not Series or DataFrame
        with pytest.raises(TypeError, match="must be pandas Series or DataFrame"):
            task.execute()

    def test_bands_with_mismatched_indices_raises_error(self):
        """Test that bands with mismatched indices raise ValueError."""
        data = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        lower = pd.Series([0.5, 1.5], index=pd.date_range("2020-01-01", periods=2))  # Wrong length
        upper = pd.Series([1.5, 2.5], index=pd.date_range("2020-01-01", periods=2))
        bands = {"confidence": (lower, upper)}

        task = TimeseriesPlotTask(data=data, bands=bands)
        with pytest.raises(ValueError, match="band indices must match"):
            task.execute()

    def test_figure_spec_includes_metadata(self):
        """Test that FigureSpec includes provided metadata."""
        data = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        task = TimeseriesPlotTask(data=data, title="Test", xlabel="Time", ylabel="Value")
        views, spec = task.execute()

        assert spec.title == "Test"
        assert spec.xlabel == "Time"
        assert spec.ylabel == "Value"


class TestBacktestPlotTask:
    """Tests for BacktestPlotTask."""

    def test_basic_backtest_returns_scatter_view(self):
        """Test that basic backtest returns ScatterView."""
        results = pd.DataFrame({"y_true": [1, 2, 3], "y_pred": [1.1, 2.1, 2.9]})
        task = BacktestPlotTask(results=results)
        views, spec = task.execute()

        assert len(views) == 1
        assert isinstance(views[0], ScatterView)
        assert isinstance(spec, FigureSpec)

    def test_with_fold_id_returns_multiple_views(self):
        """Test that fold_id creates separate views for each fold."""
        results = pd.DataFrame(
            {
                "y_true": [1, 2, 3, 4, 5],
                "y_pred": [1.1, 2.1, 2.9, 4.1, 4.9],
                "fold_id": [0, 0, 1, 1, 2],
            }
        )
        task = BacktestPlotTask(results=results, fold_id_col="fold_id")
        views, spec = task.execute()

        assert len(views) == 3  # One for each fold
        assert all(isinstance(v, ScatterView) for v in views)

    def test_missing_y_true_col_raises_error(self):
        """Test that missing y_true column raises ValueError."""
        results = pd.DataFrame({"y_pred": [1, 2, 3]})
        task = BacktestPlotTask(results=results, y_true_col="y_true")
        with pytest.raises(ValueError, match="not found in results"):
            task.execute()

    def test_missing_y_pred_col_raises_error(self):
        """Test that missing y_pred column raises ValueError."""
        results = pd.DataFrame({"y_true": [1, 2, 3]})
        task = BacktestPlotTask(results=results, y_pred_col="y_pred")
        with pytest.raises(ValueError, match="not found in results"):
            task.execute()

    def test_figure_spec_includes_metadata(self):
        """Test that FigureSpec includes provided metadata."""
        results = pd.DataFrame({"y_true": [1, 2, 3], "y_pred": [1.1, 2.1, 2.9]})
        task = BacktestPlotTask(results=results, title="Test", xlabel="True", ylabel="Pred")
        views, spec = task.execute()

        assert spec.title == "Test"
        assert spec.xlabel == "True"
        assert spec.ylabel == "Pred"

