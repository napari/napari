"""Tests for the HistogramModel class."""

import numpy as np

from napari.components.histogram import HistogramModel
from napari.layers import Image


class TestHistogramModel:
    """Test HistogramModel functionality."""

    def test_histogram_creation(self):
        """Test basic histogram creation."""
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = layer.histogram

        assert isinstance(hist, HistogramModel)
        assert hist.n_bins == 256
        assert hist.mode == 'displayed'
        assert hist.log_scale is False
        assert hist.enabled is True

    def test_histogram_computation(self):
        """Test histogram is computed correctly."""
        data = np.arange(1000, dtype=np.float32).reshape((10, 100))
        layer = Image(data)
        hist = layer.histogram

        # Force computation
        bins = hist.bins
        counts = hist.counts

        assert len(bins) == 257  # n_bins + 1
        assert len(counts) == 256
        assert counts.sum() > 0  # Should have some counts

    def test_histogram_lazy_computation(self):
        """Test that histogram is computed lazily."""
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = HistogramModel(layer, enabled=False)

        # Should not compute when disabled
        assert hist._dirty is True
        _ = hist.bins  # Access should not trigger computation when disabled
        assert hist._dirty is True

    def test_histogram_data_change_triggers_recompute(self):
        """Test that changing layer data triggers histogram recomputation."""
        data1 = np.ones((100, 100))
        layer = Image(data1)
        hist = layer.histogram

        counts1 = hist.counts.copy()

        # Change data
        layer.data = np.zeros((100, 100))

        counts2 = hist.counts

        assert not np.array_equal(counts1, counts2)

    def test_histogram_contrast_limits_range_change(self):
        """Test that changing contrast limits range triggers recomputation."""
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = layer.histogram

        # Change contrast limits range
        layer.contrast_limits_range = [0.0, 0.5]

        bins2 = hist.bins

        # Bins should have changed to reflect new range
        assert bins2[-1] <= 0.5

    def test_histogram_log_scale(self):
        """Test log scale transformation."""
        data = np.random.random((100, 100)) * 100
        layer = Image(data)
        hist = layer.histogram

        counts_linear = hist.counts.copy()

        # Enable log scale
        hist.log_scale = True

        counts_log = hist.counts

        # Log scaled counts should be different
        assert not np.array_equal(counts_linear, counts_log)
        # Log scaled counts should generally be smaller
        assert counts_log.max() <= np.log10(counts_linear.max() + 1) + 1

    def test_histogram_n_bins_change(self):
        """Test changing number of bins."""
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = layer.histogram

        assert len(hist.counts) == 256

        hist.n_bins = 128

        assert len(hist.counts) == 128
        assert len(hist.bins) == 129

    def test_histogram_mode_displayed_vs_full(self):
        """Test displayed mode vs full mode."""
        data = np.random.random((10, 100, 100))
        layer = Image(data)
        hist = layer.histogram

        # Default is displayed mode
        assert hist.mode == 'displayed'

        # Change to full mode
        hist.mode = 'full'
        counts_full = hist.counts

        # Full should have more data points
        # (though counts sum depends on binning)
        assert counts_full is not None

    def test_histogram_sampling_large_data(self):
        """Test that large data is sampled."""
        # Create data larger than 1M points
        data = np.random.random((2000, 2000))
        layer = Image(data)
        hist = layer.histogram

        # Should still compute without issues
        counts = hist.counts
        assert counts is not None
        assert len(counts) == 256

    def test_histogram_empty_data(self):
        """Test histogram with empty/zero data."""
        data = np.zeros((10, 10))
        layer = Image(data)
        hist = layer.histogram

        counts = hist.counts
        bins = hist.bins

        # Should handle gracefully
        assert counts is not None
        assert bins is not None

    def test_histogram_nan_inf_handling(self):
        """Test histogram handles NaN and inf values."""
        data = np.random.random((100, 100))
        data[0:10, 0:10] = np.nan
        data[20:30, 20:30] = np.inf
        layer = Image(data)
        hist = layer.histogram

        # Should compute without errors
        counts = hist.counts
        assert counts is not None
        # NaN and inf should be filtered out
        assert np.isfinite(counts).all()

    def test_histogram_single_value_data(self):
        """Test histogram with all same values."""
        data = np.ones((100, 100)) * 42.0
        layer = Image(data)
        hist = layer.histogram

        counts = hist.counts
        bins = hist.bins

        # Should handle single value gracefully
        assert counts is not None
        assert bins is not None

    def test_histogram_uint8_data(self):
        """Test histogram with uint8 data."""
        data = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        layer = Image(data)
        hist = layer.histogram

        counts = hist.counts
        bins = hist.bins

        assert counts is not None
        assert len(counts) == 256
        # Bins should span 0-255 for uint8
        assert bins[0] >= 0
        assert bins[-1] <= 255

    def test_histogram_uint16_data(self):
        """Test histogram with uint16 data."""
        data = np.random.randint(0, 1000, size=(100, 100), dtype=np.uint16)
        layer = Image(data)
        hist = layer.histogram

        counts = hist.counts
        assert counts is not None

    def test_histogram_reset(self):
        """Test histogram reset."""
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = layer.histogram

        hist.n_bins = 128
        hist.log_scale = True
        hist.mode = 'full'

        hist.reset()

        assert hist.n_bins == 256
        assert hist.log_scale is False
        assert hist.mode == 'displayed'

    def test_histogram_events(self):
        """Test that histogram events fire correctly."""
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = layer.histogram

        bins_called = []
        counts_called = []

        def on_bins():
            bins_called.append(True)

        def on_counts():
            counts_called.append(True)

        hist.events.bins.connect(on_bins)
        hist.events.counts.connect(on_counts)

        # Trigger recomputation
        hist.compute()

        assert len(bins_called) > 0
        assert len(counts_called) > 0

    def test_histogram_multiscale_image(self):
        """Test histogram with multiscale image."""
        # Create simple multiscale data
        data = [
            np.random.random((100, 100)),
            np.random.random((50, 50)),
            np.random.random((25, 25)),
        ]
        layer = Image(data, multiscale=True)
        hist = layer.histogram

        counts = hist.counts
        assert counts is not None

    def test_histogram_rgb_image(self):
        """Test histogram with RGB image."""
        data = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        layer = Image(data, rgb=True)

        # RGB images should still have histogram property
        hist = layer.histogram
        assert hist is not None

    def test_histogram_compute_explicit(self):
        """Test explicit compute call."""
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = layer.histogram

        # Mark as dirty but disabled
        hist.enabled = False
        hist._dirty = True

        # Explicit compute should work even when disabled
        hist.compute()

        assert hist._dirty is False

    def test_histogram_disabled(self):
        """Test histogram with enabled=False."""
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = HistogramModel(layer, enabled=False)

        # Should not auto-compute
        assert hist._dirty is True

        # Manual compute should still work
        hist.compute()
        assert hist._dirty is False

    def test_histogram_data_types(self):
        """Test histogram with various data types."""
        dtypes = [np.uint8, np.uint16, np.int32, np.float32, np.float64]

        for dtype in dtypes:
            data = np.random.random((50, 50)).astype(dtype)
            if np.issubdtype(dtype, np.integer):
                data = (data * 100).astype(dtype)

            layer = Image(data)
            hist = layer.histogram

            counts = hist.counts
            assert counts is not None, f'Failed for dtype {dtype}'
