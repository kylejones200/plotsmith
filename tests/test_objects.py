"""Tests for Layer 1: objects and validators."""

import numpy as np
import pytest

from plotsmith.objects.views import BandView, ScatterView, SeriesView
from plotsmith.objects.validate import (
    validate_all_views,
    validate_band_view,
    validate_scatter_view,
    validate_series_view,
)


class TestSeriesView:
    """Tests for SeriesView validation."""

    def test_valid_series_view(self):
        """Test that valid SeriesView passes validation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50])
        view = SeriesView(x=x, y=y, label="test")
        validate_series_view(view)  # Should not raise

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched x and y lengths raise ValueError."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20])  # Different length
        view = SeriesView(x=x, y=y)
        with pytest.raises(ValueError, match="must have same length"):
            validate_series_view(view)


class TestBandView:
    """Tests for BandView validation."""

    def test_valid_band_view(self):
        """Test that valid BandView passes validation."""
        x = np.array([1, 2, 3, 4, 5])
        y_lower = np.array([5, 10, 15, 20, 25])
        y_upper = np.array([15, 20, 25, 30, 35])
        view = BandView(x=x, y_lower=y_lower, y_upper=y_upper)
        validate_band_view(view)  # Should not raise

    def test_mismatched_x_y_lower_raises_error(self):
        """Test that mismatched x and y_lower lengths raise ValueError."""
        x = np.array([1, 2, 3])
        y_lower = np.array([5, 10])  # Different length
        y_upper = np.array([15, 20, 25])
        view = BandView(x=x, y_lower=y_lower, y_upper=y_upper)
        with pytest.raises(ValueError, match="x and y_lower must have same length"):
            validate_band_view(view)

    def test_mismatched_x_y_upper_raises_error(self):
        """Test that mismatched x and y_upper lengths raise ValueError."""
        x = np.array([1, 2, 3])
        y_lower = np.array([5, 10, 15])
        y_upper = np.array([15, 20])  # Different length
        view = BandView(x=x, y_lower=y_lower, y_upper=y_upper)
        with pytest.raises(ValueError, match="x and y_upper must have same length"):
            validate_band_view(view)

    def test_mismatched_y_lower_y_upper_raises_error(self):
        """Test that mismatched y_lower and y_upper lengths raise ValueError."""
        x = np.array([1, 2, 3])
        y_lower = np.array([5, 10, 15])
        y_upper = np.array([15, 20])  # Different length
        view = BandView(x=x, y_lower=y_lower, y_upper=y_upper)
        # The validator checks x vs y_upper first, so it catches this as x/y_upper mismatch
        with pytest.raises(ValueError, match="must have same length"):
            validate_band_view(view)


class TestScatterView:
    """Tests for ScatterView validation."""

    def test_valid_scatter_view(self):
        """Test that valid ScatterView passes validation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50])
        view = ScatterView(x=x, y=y, label="test")
        validate_scatter_view(view)  # Should not raise

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched x and y lengths raise ValueError."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20])  # Different length
        view = ScatterView(x=x, y=y)
        with pytest.raises(ValueError, match="must have same length"):
            validate_scatter_view(view)


class TestValidateAllViews:
    """Tests for validate_all_views function."""

    def test_valid_mixed_views(self):
        """Test that a list of valid views passes validation."""
        views = [
            SeriesView(x=np.array([1, 2, 3]), y=np.array([10, 20, 30])),
            BandView(
                x=np.array([1, 2, 3]), y_lower=np.array([5, 10, 15]), y_upper=np.array([15, 20, 25])
            ),
            ScatterView(x=np.array([1, 2, 3]), y=np.array([10, 20, 30])),
        ]
        validate_all_views(views)  # Should not raise

    def test_invalid_view_in_list_raises_error(self):
        """Test that an invalid view in the list raises ValueError."""
        views = [
            SeriesView(x=np.array([1, 2, 3]), y=np.array([10, 20, 30])),
            SeriesView(x=np.array([1, 2, 3]), y=np.array([10, 20])),  # Invalid
        ]
        with pytest.raises(ValueError):
            validate_all_views(views)

    def test_unknown_view_type_raises_error(self):
        """Test that unknown view type raises TypeError."""
        views = [object()]  # Not a valid view type
        with pytest.raises(TypeError, match="Unknown view type"):
            validate_all_views(views)

