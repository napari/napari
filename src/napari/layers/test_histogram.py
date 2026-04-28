"""Tests for the HistogramModel class."""

from dataclasses import replace

import numpy as np

from napari.layers import Image
from napari.layers._histogram import HistogramModel


class TestHistogramModel:
    """Test HistogramModel functionality."""

    def test_histogram_creation(self):
        """Test basic histogram creation."""
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = layer.histogram

        assert isinstance(hist, HistogramModel)
        assert hist.n_bins == 256
        assert hist.mode == 'canvas'
        assert hist.log_scale is False
        # enabled defaults to False; histogram is inert until a widget
        # explicitly enables it (popup shown or inline histogram toggled on).
        assert hist.enabled is False

    def test_histogram_computation(self):
        """Test histogram is computed on first access of bins/counts."""
        data = np.arange(1000, dtype=np.float32).reshape((10, 100))
        layer = Image(data)
        hist = layer.histogram

        # Computation is always triggered by accessing bins/counts when dirty.
        bins = hist.bins
        counts = hist.counts

        assert len(bins) == 257  # n_bins + 1
        assert len(counts) == 256
        assert counts.sum() > 0

    def test_histogram_not_computed_on_creation(self):
        """Histogram must not run np.histogram during model creation.

        Computation is deferred until the histogram widget is shown or the
        caller explicitly accesses bins/counts. enabled=False (the default)
        guarantees no event-driven compute fires during or after __init__.
        """
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = HistogramModel(layer)  # enabled=False by default

        assert hist._dirty is True  # computation has not run

    def test_histogram_disabled_prevents_event_driven_recompute(self):
        """With enabled=False, data-change events do not trigger auto-recompute.

        Explicit access to bins/counts still computes regardless of enabled.

        NOTE: always use ``layer.histogram`` (never a standalone
        ``HistogramModel(layer)``).  Pydantic compares model instances by
        field values, so two models with identical settings are considered
        equal by napari's EventEmitter, which silently drops the second
        connection as a duplicate.  ``layer.histogram`` is registered first and
        is therefore the canonical connected instance.
        """
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = layer.histogram  # enabled=False by default; properly connected

        # Starts dirty; no event-driven compute has run yet.
        assert hist._dirty is True

        # Explicit access always computes when dirty.
        _ = hist.bins
        assert hist._dirty is False

        # A data change must mark dirty but NOT auto-recompute (enabled=False).
        layer.data = np.random.random((100, 100))
        assert hist._dirty is True  # dirty again, but no event-driven compute

    def test_histogram_enabled_triggers_immediate_compute(self):
        """Flipping enabled to True computes immediately if data is dirty."""
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = layer.histogram  # enabled=False by default; dirty=True

        assert hist._dirty is True
        hist.enabled = True  # _on_enabled_change fires compute

        assert hist._dirty is False

    def test_histogram_data_change_triggers_recompute(self):
        """Test that changing layer data results in fresh counts on access."""
        data1 = np.ones((100, 100))
        layer = Image(data1)
        hist = layer.histogram

        counts1 = hist.counts.copy()  # explicit access → compute

        layer.data = np.zeros(
            (100, 100)
        )  # marks dirty (enabled=False → no auto-compute)
        counts2 = hist.counts  # explicit access → recompute with new data

        assert not np.array_equal(counts1, counts2)

    def test_histogram_contrast_limits_range_change(self):
        """Test that changing contrast limits range triggers recomputation."""
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = layer.histogram

        layer.contrast_limits_range = [0.0, 0.5]

        bins2 = hist.bins

        assert bins2[-1] <= 0.5

    def test_histogram_log_scale(self):
        """Test log scale transformation."""
        data = np.random.random((100, 100)) * 100
        layer = Image(data)
        hist = layer.histogram

        counts_linear = hist.counts.copy()

        hist.log_scale = True

        counts_log = hist.counts

        assert not np.array_equal(counts_linear, counts_log)
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

        assert hist.mode == 'canvas'

        hist.mode = 'full'
        counts_full = hist.counts

        assert counts_full is not None

    def test_histogram_canvas_mode_falls_back_before_first_real_slice(self):
        """Canvas mode should ignore the placeholder 1x1 sliced sample."""
        data = np.arange(64, dtype=np.uint8).reshape((8, 8))
        layer = Image(data)
        layer._slicing_state._slice = replace(
            layer._slice,
            image=replace(
                layer._slice.image, raw=data[:1, :1], view=data[:1, :1]
            ),
        )
        hist = HistogramModel(layer)

        assert hist.mode == 'canvas'
        assert hist.counts.sum() == data.size

        hist.mode = 'full'
        assert hist.counts.sum() == data.size

    def test_histogram_canvas_mode_falls_back_before_first_real_rgb_slice(
        self,
    ):
        """Canvas mode should ignore the placeholder RGB sample before slicing."""
        data = np.arange(8 * 8 * 3, dtype=np.uint8).reshape((8, 8, 3))
        layer = Image(data, rgb=True)
        layer._slicing_state._slice = replace(
            layer._slice,
            image=replace(
                layer._slice.image,
                raw=data[:1, :1, :],
                view=data[:1, :1, :],
            ),
        )
        hist = HistogramModel(layer)

        assert hist.mode == 'canvas'
        # After luminance conversion, each pixel yields one value regardless
        # of channel count. The number of samples equals the pixel count.
        n_pixels = data.shape[0] * data.shape[1]
        assert hist.counts.sum() == n_pixels

        hist.mode = 'full'
        assert hist.counts.sum() == n_pixels

    def test_histogram_sampling_large_data(self):
        """Test that large data is sampled."""
        data = np.random.random((2000, 2000))
        layer = Image(data)
        hist = layer.histogram

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

        assert counts is not None
        assert bins is not None

    def test_histogram_nan_inf_handling(self):
        """Test histogram handles NaN and inf values."""
        data = np.random.random((100, 100))
        data[0:10, 0:10] = np.nan
        data[20:30, 20:30] = np.inf
        layer = Image(data)
        hist = layer.histogram

        counts = hist.counts
        assert counts is not None
        assert np.isfinite(counts).all()

    def test_histogram_single_value_data(self):
        """Test histogram with all same values."""
        data = np.ones((100, 100)) * 42.0
        layer = Image(data)
        hist = layer.histogram

        counts = hist.counts
        bins = hist.bins

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
        assert hist.mode == 'canvas'

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

        hist.compute()

        assert len(bins_called) > 0
        assert len(counts_called) > 0

    def test_histogram_multiscale_image(self):
        """Test histogram with multiscale image."""
        data = [
            np.random.random((100, 100)),
            np.random.random((50, 50)),
            np.random.random((25, 25)),
        ]
        layer = Image(data, multiscale=True)
        hist = layer.histogram

        counts = hist.counts
        assert counts is not None

    def test_histogram_multiscale_full_uses_coarsest_level(self):
        """Full mode should use the coarsest multiscale level."""
        data = [
            np.arange(10000, dtype=np.float32).reshape((100, 100)),
            np.arange(2500, dtype=np.float32).reshape((50, 50)),
            np.arange(625, dtype=np.float32).reshape((25, 25)),
        ]
        layer = Image(data, multiscale=True)
        hist = layer.histogram

        hist.mode = 'full'

        assert hist.counts.sum() == data[-1].size

    def test_histogram_rgb_image(self):
        """Test histogram with RGB image."""
        data = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        layer = Image(data, rgb=True)

        hist = layer.histogram
        assert hist is not None

    def test_histogram_compute_explicit(self):
        """Test explicit compute call works regardless of enabled state."""
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = layer.histogram

        hist.enabled = False
        hist._dirty = True

        hist.compute()

        assert hist._dirty is False

    def test_histogram_disabled(self):
        """Test histogram with enabled=False."""
        data = np.random.random((100, 100))
        layer = Image(data)
        hist = HistogramModel(layer, enabled=False)

        assert hist._dirty is True

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
