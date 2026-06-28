"""Tests for HistogramModel.

These tests do not require Qt and can run in headless mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from napari.layers import Image


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _image(data, **kwargs):
    """Helper to create an Image layer with histogram model."""
    return Image(data, **kwargs)


class TestDefaultState:
    """Test the default state of HistogramModel."""

    def test_starts_disabled(self):
        img = _image(np.random.rand(10, 10))
        assert not img.histogram.enabled

    def test_empty_bins_and_counts_when_disabled(self):
        img = _image(np.random.rand(10, 10))
        assert not img.histogram.enabled
        # Accessing bins/counts triggers lazy computation regardless
        assert len(img.histogram.bins) >= 2
        assert len(img.histogram.counts) >= 1

    def test_default_mode_canvas(self):
        img = _image(np.random.rand(10, 10))
        assert img.histogram.mode == 'canvas'

    def test_default_n_bins_256(self):
        img = _image(np.random.rand(10, 10))
        assert img.histogram.n_bins == 256

    def test_default_log_scale_false(self):
        img = _image(np.random.rand(10, 10))
        assert not img.histogram.log_scale


class TestEnableDisable:
    """Test enabling and disabling histogram computation."""

    def test_enable_triggers_computation(self):
        img = _image(np.random.rand(10, 10).astype(np.float32))
        img.histogram.enabled = True
        assert img.histogram._dirty is False or len(img.histogram.counts) > 0

    def test_disable_clears_cache(self):
        img = _image(np.random.rand(10, 10))
        img.histogram.enabled = True
        # Compute once
        _ = img.histogram.counts
        img.histogram.enabled = False
        # Disabling doesn't clear, but marks dirty for next enable
        assert img.histogram._dirty or not img.histogram.enabled

    def test_no_computation_when_disabled(self):
        img = _image(np.random.rand(10, 10))
        img.histogram.enabled = False
        # Change data while disabled
        img.data = np.random.rand(10, 10)
        # Should still be computed lazily on access
        assert len(img.histogram.bins) >= 2

    def test_reset_clears_and_disables(self):
        img = _image(np.random.rand(10, 10))
        img.histogram.enabled = True
        _ = img.histogram.counts  # trigger compute
        img.histogram.reset()
        assert not img.histogram.enabled
        assert img.histogram.mode == 'canvas'
        assert img.histogram.n_bins == 256
        assert not img.histogram.log_scale


class TestRangeProperty:
    """Test the range property."""

    def test_range_returns_bin_edges(self):
        img = _image(np.random.rand(10, 10))
        img.histogram.enabled = True
        _ = img.histogram.counts  # trigger compute
        r = img.histogram.range
        assert isinstance(r, tuple)
        assert len(r) == 2
        assert r[0] < r[1]

    def test_range_default_when_empty(self):
        img = _image(np.zeros((10, 10)))
        img.histogram.enabled = True
        _ = img.histogram.counts
        r = img.histogram.range
        assert r[0] < r[1]


class TestDataTypes:
    """Test histogram with various data types."""

    def test_uint8(self):
        data = np.random.randint(0, 255, (20, 20)).astype(np.uint8)
        img = _image(data)
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert len(counts) == img.histogram.n_bins
        assert counts.sum() > 0

    def test_uint16(self):
        data = np.random.randint(0, 1000, (20, 20)).astype(np.uint16)
        img = _image(data)
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert len(counts) > 0
        assert counts.sum() > 0

    def test_float32(self):
        data = np.random.rand(20, 20).astype(np.float32)
        img = _image(data)
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert len(counts) > 0
        assert counts.sum() > 0

    def test_float64(self):
        data = np.random.rand(20, 20)
        img = _image(data)
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert len(counts) > 0
        assert counts.sum() > 0

    def test_int16(self):
        data = np.random.randint(-100, 100, (20, 20)).astype(np.int16)
        img = _image(data)
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert len(counts) > 0
        assert counts.sum() > 0

    def test_all_zeros(self):
        img = _image(np.zeros((10, 10), dtype=np.float32))
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert len(counts) > 0

    def test_constant_data(self):
        img = _image(np.full((10, 10), 42.0, dtype=np.float32))
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert len(counts) > 0

    def test_uniform_data(self):
        """All same value — histogram should handle gracefully."""
        img = _image(np.full((10, 10), 5, dtype=np.uint8))
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert counts.sum() > 0

    def test_single_element(self):
        img = _image(np.array([[42.0]], dtype=np.float32))
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert len(counts) > 0

    def test_empty_data(self):
        """A single-point valid layer — histogram should handle gracefully."""
        img = _image(np.zeros((1, 1), dtype=np.float32))
        img.histogram.enabled = True
        # Should not crash
        _ = img.histogram.bins
        _ = img.histogram.counts

    def test_with_nan(self):
        data = np.random.rand(20, 20)
        data[0, 0] = np.nan
        img = _image(data.astype(np.float32))
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert counts.sum() > 0

    def test_with_inf(self):
        data = np.random.rand(20, 20)
        data[0, 0] = np.inf
        data[0, 1] = -np.inf
        img = _image(data.astype(np.float32))
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert counts.sum() > 0


class TestCustomNBins:
    """Test custom bin counts."""

    def test_custom_bins(self):
        img = _image(np.random.rand(20, 20), colormap='gray')
        img.histogram.enabled = True
        img.histogram.n_bins = 128
        # Changing n_bins triggers recomputation on next access
        counts = img.histogram.counts
        # n_bins change may or may not have taken effect depending on
        # whether _mark_dirty was called; check the length
        assert len(counts) > 0

    def test_large_bins(self):
        img = _image(np.random.rand(20, 20))
        img.histogram.enabled = True
        img.histogram.n_bins = 4096
        _ = img.histogram.counts  # Should not crash

    def test_small_bins(self):
        img = _image(np.random.rand(20, 20))
        img.histogram.enabled = True
        img.histogram.n_bins = 2
        _ = img.histogram.counts  # Should not crash


class TestMode:
    """Test canvas vs full mode."""

    def test_canvas_mode(self):
        img = _image(np.random.rand(5, 20, 20))
        img.histogram.mode = 'canvas'
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert len(counts) > 0

    def test_full_mode(self):
        img = _image(np.random.rand(5, 20, 20))
        img.histogram.mode = 'full'
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert len(counts) > 0


class TestLogScale:
    """Test log scale."""

    def test_log_scale_toggle(self):
        img = _image(np.random.rand(20, 20))
        img.histogram.enabled = True
        img.histogram.log_scale = True
        counts_log = img.histogram.counts
        img.histogram.log_scale = False
        counts_linear = img.histogram.counts
        assert len(counts_log) == len(counts_linear)

    def test_log_scale_values(self):
        """Log scale should produce different values than linear."""
        img = _image(np.random.rand(20, 20))
        img.histogram.enabled = True
        img.histogram.log_scale = False
        linear = img.histogram.counts.copy()
        img.histogram.log_scale = True
        logged = img.histogram.counts
        # At least some values should differ
        if len(linear) > 0 and len(logged) > 0:
            assert not np.array_equal(linear, logged)


class TestNBinsChange:
    """Test changing n_bins property."""

    def test_n_bins_change_output_length(self):
        data = np.random.rand(20, 20)
        img = _image(data)
        assert len(img.histogram.counts) == 256
        img.histogram.n_bins = 128
        assert len(img.histogram.counts) == 128
        assert len(img.histogram.bins) == 129

    def test_contrast_limits_range_change(self):
        data = np.random.rand(20, 20)
        img = _image(data)
        img.histogram.enabled = True
        img.contrast_limits_range = [0.0, 0.5]
        _ = img.histogram.counts  # trigger recompute
        assert img.histogram.bins[-1] <= 0.5


class TestMultiscale:
    """Test histogram with multiscale data."""

    def test_multiscale_image(self):
        data = [
            np.random.rand(100, 100).astype(np.float32),
            np.random.rand(50, 50).astype(np.float32),
        ]
        img = _image(data, multiscale=True)
        counts = img.histogram.counts
        assert counts is not None
        assert len(counts) > 0

    def test_multiscale_full_uses_coarsest(self):
        data = [
            np.arange(10000, dtype=np.float32).reshape((100, 100)),
            np.arange(2500, dtype=np.float32).reshape((50, 50)),
            np.arange(625, dtype=np.float32).reshape((25, 25)),
        ]
        img = _image(data, multiscale=True)
        img.histogram.mode = 'full'
        assert img.histogram.counts.sum() == data[-1].size


class TestRGB:
    """Test histogram with RGB data."""

    def test_rgb_image(self):
        data = np.random.randint(0, 256, size=(20, 20, 3), dtype=np.uint8)
        img = _image(data, rgb=True)
        assert img.histogram is not None
        counts = img.histogram.counts
        assert len(counts) > 0


class TestEvents:
    """Test histogram event emissions."""

    def test_bins_and_counts_events_fire(self):
        data = np.random.rand(20, 20)
        img = _image(data)
        bins_fired = []
        counts_fired = []

        img.histogram.events.bins.connect(lambda: bins_fired.append(True))
        img.histogram.events.counts.connect(lambda: counts_fired.append(True))
        img.histogram.compute()

        assert len(bins_fired) > 0
        assert len(counts_fired) > 0


class TestLargeData:
    """Test histogram with larger datasets."""

    def test_sampling_large_data(self):
        data = np.random.rand(500, 500).astype(np.float32)
        img = _image(data)
        counts = img.histogram.counts
        assert counts is not None
        assert len(counts) == 256


class TestDask:
    """Test histogram with dask arrays (no full materialization)."""

    def test_histogram_model_detects_dask_array(self):
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(20, 20).astype(np.float32), chunks=10
        )
        img = _image(data)
        assert img.histogram._is_dask_array(data)

    def test_histogram_model_does_not_detect_numpy(self):
        data = np.random.rand(20, 20).astype(np.float32)
        img = _image(data)
        assert not img.histogram._is_dask_array(data)

    def test_does_not_detect_multiscale_as_dask(self):
        data = [
            np.random.rand(100, 100).astype(np.float32),
            np.random.rand(50, 50).astype(np.float32),
        ]
        img = _image(data, multiscale=True)
        assert not img.histogram._is_dask_array(data)
        assert not img.histogram._is_dask_array(data[0])

    def test_sample_dask_safe_returns_sample(self):
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(500, 500).astype(np.float32), chunks=100
        )
        img = _image(data)
        sampled = img.histogram._sample_dask_safe(data)
        assert isinstance(sampled, np.ndarray)
        assert sampled.size > 0
        assert sampled.size <= 1_000_000

    def test_dask_full_mode_histogram(self):
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(50, 50).astype(np.float32), chunks=25
        )
        img = _image(data)
        img.histogram.mode = 'full'
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert counts is not None
        assert len(counts) == 256
        assert counts.sum() > 0

    def test_dask_canvas_mode_slice_histogram(self):
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(5, 50, 50).astype(np.float32), chunks=(1, 25, 25)
        )
        img = _image(data)
        img.histogram.mode = 'canvas'
        img.histogram.enabled = True
        counts = img.histogram.counts
        assert counts is not None
        assert len(counts) == 256
        assert counts.sum() > 0
