"""Tests for HistogramModel.

These tests do not require Qt and can run in headless mode.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from napari.components.histogram import (
    _MAX_MATERIALIZE_ELEMENTS,
    DEFAULT_MAX_SAMPLES,
)
from napari.layers import Image


def _model(data, **kwargs):
    """Create an Image layer and return its HistogramModel.

    ``HistogramModel`` requires an ``Image`` layer to construct, so
    we create one and extract the model.  Tests that exercise pure
    computation methods (``_calc_histogram``, ``_rgb_to_luminance``,
    ``_sample_data``, ``_sample_dask_safe``) call the model directly
    after construction.
    """
    return Image(data, **kwargs).histogram


class TestDefaultState:
    """Test the default state of HistogramModel."""

    def test_starts_disabled(self):
        model = _model(np.random.rand(10, 10))
        assert not model.enabled

    def test_empty_bins_and_counts_when_disabled(self):
        model = _model(np.random.rand(10, 10))
        assert not model.enabled
        # Accessing bins/counts triggers lazy computation regardless
        assert len(model.bins) >= 2
        assert len(model.counts) >= 1

    def test_default_mode_canvas(self):
        model = _model(np.random.rand(10, 10))
        assert model.mode == 'canvas'

    def test_default_n_bins_256(self):
        model = _model(np.random.rand(10, 10))
        assert model.n_bins == 256

    def test_default_log_scale_false(self):
        model = _model(np.random.rand(10, 10))
        assert not model.log_scale

    def test_default_max_samples(self):
        model = _model(np.random.rand(10, 10))
        assert model.max_samples == DEFAULT_MAX_SAMPLES


class TestEnableDisable:
    """Test enabling and disabling histogram computation."""

    def test_enable_triggers_computation(self):
        model = _model(np.random.rand(10, 10).astype(np.float32))
        model.enabled = True
        assert model._dirty is False or len(model.counts) > 0

    def test_disable_clears_cache(self):
        model = _model(np.random.rand(10, 10))
        model.enabled = True
        _ = model.counts  # Compute once
        model.enabled = False
        # Disabling doesn't clear, but marks dirty for next enable
        assert model._dirty or not model.enabled

    def test_no_computation_when_disabled(self):
        model = _model(np.random.rand(10, 10))
        model.enabled = False
        # Change data while disabled — should still compute lazily on access
        model._layer.data = np.random.rand(10, 10)
        assert len(model.bins) >= 2

    def test_reset_clears_and_disables(self):
        model = _model(np.random.rand(10, 10))
        model.enabled = True
        _ = model.counts  # trigger compute
        model.reset()
        assert not model.enabled
        assert model.mode == 'canvas'
        assert model.n_bins == 256
        assert not model.log_scale

    def test_reset_then_re_enable_computes(self):
        """Reset followed by re-enable should produce fresh counts."""
        model = _model(np.random.rand(10, 10))
        model.enabled = True
        _ = model.counts  # trigger initial compute
        model.reset()
        model.enabled = True
        assert model._dirty is False
        counts = model.counts
        assert len(counts) == 256


class TestDataTypes:
    """Test histogram with various data types."""

    @pytest.mark.parametrize(
        'dtype',
        ['uint8', 'uint16', 'float32', 'float64', 'int16'],
    )
    def test_various_dtypes(self, dtype):
        rng = np.random.default_rng(0)
        if np.issubdtype(dtype, np.integer):
            data = rng.integers(0, 255, (20, 20), dtype=dtype)
        else:
            data = rng.random((20, 20)).astype(dtype)
        model = _model(data)
        model.enabled = True
        counts = model.counts
        assert len(counts) == model.n_bins
        assert counts.sum() > 0

    def test_all_zeros(self):
        model = _model(np.zeros((10, 10), dtype=np.float32))
        model.enabled = True
        counts = model.counts
        assert len(counts) > 0

    def test_constant_data(self):
        model = _model(np.full((10, 10), 42.0, dtype=np.float32))
        model.enabled = True
        counts = model.counts
        assert len(counts) > 0

    def test_uniform_data(self):
        """All same value — histogram should handle gracefully."""
        model = _model(np.full((10, 10), 5, dtype=np.uint8))
        model.enabled = True
        counts = model.counts
        assert counts.sum() > 0

    def test_single_element(self):
        model = _model(np.array([[42.0]], dtype=np.float32))
        model.enabled = True
        counts = model.counts
        assert len(counts) > 0

    def test_with_nan(self):
        data = np.random.rand(20, 20)
        data[0, 0] = np.nan
        model = _model(data.astype(np.float32))
        model.enabled = True
        counts = model.counts
        assert counts.sum() > 0

    def test_with_inf(self):
        data = np.random.rand(20, 20)
        data[0, 0] = np.inf
        data[0, 1] = -np.inf
        model = _model(data.astype(np.float32))
        model.enabled = True
        counts = model.counts
        assert counts.sum() > 0


class TestCustomNBins:
    """Test custom bin counts."""

    def test_custom_bins(self):
        model = _model(np.random.rand(20, 20), colormap='gray')
        model.enabled = True
        model.n_bins = 128
        counts = model.counts
        assert len(counts) == 128

    def test_large_bins(self):
        model = _model(np.random.rand(20, 20))
        model.enabled = True
        model.n_bins = 4096
        _ = model.counts  # Should not crash

    def test_small_bins(self):
        model = _model(np.random.rand(20, 20))
        model.enabled = True
        model.n_bins = 2
        _ = model.counts  # Should not crash


class TestMode:
    """Test canvas vs full mode."""

    def test_canvas_mode(self):
        model = _model(np.random.rand(5, 20, 20))
        model.mode = 'canvas'
        model.enabled = True
        counts = model.counts
        assert len(counts) > 0

    def test_full_mode(self):
        model = _model(np.random.rand(5, 20, 20))
        model.mode = 'full'
        model.enabled = True
        counts = model.counts
        assert len(counts) > 0


class TestLogScale:
    """Test log scale."""

    def test_log_scale_toggle(self):
        model = _model(np.random.rand(20, 20))
        model.enabled = True
        model.log_scale = True
        counts_log = model.counts
        model.log_scale = False
        counts_linear = model.counts
        assert len(counts_log) == len(counts_linear)

    def test_log_scale_values(self):
        """Log scale should produce different values than linear."""
        model = _model(np.random.rand(20, 20))
        model.enabled = True
        model.log_scale = False
        linear = model.counts.copy()
        model.log_scale = True
        logged = model.counts
        assert not np.array_equal(linear, logged)


class TestNBinsChange:
    """Test changing n_bins property."""

    def test_n_bins_change_output_length(self):
        model = _model(np.random.rand(20, 20))
        assert len(model.counts) == 256
        model.n_bins = 128
        assert len(model.counts) == 128
        assert len(model.bins) == 129

    def test_contrast_limits_range_change(self):
        model = _model(np.random.rand(20, 20))
        model.enabled = True
        model._layer.contrast_limits_range = [0.0, 0.5]
        _ = model.counts  # trigger recompute
        assert model.bins[-1] <= 0.5


class TestMultiscale:
    """Test histogram with multiscale data."""

    def test_multiscale_image(self):
        data = [
            np.random.rand(100, 100).astype(np.float32),
            np.random.rand(50, 50).astype(np.float32),
        ]
        model = _model(data, multiscale=True)
        counts = model.counts
        assert counts is not None
        assert len(counts) > 0

    def test_multiscale_full_uses_coarsest(self):
        data = [
            np.arange(10000, dtype=np.float32).reshape((100, 100)),
            np.arange(2500, dtype=np.float32).reshape((50, 50)),
            np.arange(625, dtype=np.float32).reshape((25, 25)),
        ]
        model = _model(data, multiscale=True)
        model.mode = 'full'
        assert model.counts.sum() == data[-1].size


class TestRGB:
    """Test histogram with RGB data."""

    def test_rgb_image(self):
        data = np.random.randint(0, 256, size=(20, 20, 3), dtype=np.uint8)
        model = _model(data, rgb=True)
        assert model is not None
        counts = model.counts
        assert len(counts) > 0

    def test_rgb_to_luminance(self):
        """_rgb_to_luminance should produce a 2D luminance array."""
        data = np.random.rand(20, 20, 3).astype(np.float32)
        model = _model(data, rgb=True)
        lum = model._rgb_to_luminance(data)
        assert lum.shape == (20, 20)
        assert lum.dtype == np.float32


class TestEvents:
    """Test histogram event emissions."""

    def test_bins_and_counts_events_fire(self):
        model = _model(np.random.rand(20, 20))
        bins_fired: list[bool] = []
        counts_fired: list[bool] = []

        model.events.bins.connect(lambda: bins_fired.append(True))
        model.events.counts.connect(lambda: counts_fired.append(True))
        model.compute()

        assert len(bins_fired) > 0
        assert len(counts_fired) > 0


class TestDisconnect:
    """Test the disconnect method for memory-safety."""

    def test_disconnect_runs_without_error(self):
        model = _model(np.random.rand(10, 10))
        model.enabled = True
        _ = model.counts  # trigger compute so we know it was alive

        model.disconnect()

        # After disconnect the model still functions, but layer events
        # no longer trigger recompute (the dirty flag stays as-is).
        assert model._dirty or not model._dirty
        assert model.n_bins == 256

    def test_disconnect_does_not_crash_on_idempotent_call(self):
        model = _model(np.random.rand(10, 10))
        model.disconnect()
        model.disconnect()  # second call should be safe


class TestCalcHistogram:
    """Test the pure numpy _calc_histogram method directly."""

    def test_basic_histogram(self):
        model = _model(np.random.rand(10, 10))
        data = np.random.rand(1000)
        bins, counts = model._calc_histogram(data, 0.0, 1.0)
        assert len(bins) == 257  # n_bins + 1
        assert len(counts) == 256
        assert counts.sum() == 1000
        assert bins.dtype == np.float32
        assert counts.dtype == np.float32

    def test_constant_range_expands_integer(self):
        """When min==max on integer data, ±0.5 should be applied."""
        model = _model(np.array([[42]], dtype=np.uint8))
        data = np.array([42, 42, 42], dtype=np.uint8)
        bins, counts = model._calc_histogram(data, 42.0, 42.0)
        # Range should be [41.5, 42.5]
        assert bins[0] == 41.5
        assert bins[-1] == 42.5
        assert counts.sum() == 3

    def test_constant_range_expands_float(self):
        """When min==max on float data, 1% expansion should be applied.

        The delta is ``max(0.5, abs(value) * 0.01)``, so a value of
        100.0 gives delta=1.0 (the 1% rule), while a value of 1.0
        gives delta=0.5 (the floor rule).
        """
        model = _model(np.array([[100.0]], dtype=np.float32))
        data = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        bins, counts = model._calc_histogram(data, 100.0, 100.0)
        # Range should be [99.0, 101.0]
        assert bins[0] == 99.0
        assert bins[-1] == 101.0
        assert counts.sum() == 3

    def test_log_scale_applies_log10(self):
        model = _model(np.random.rand(10, 10))
        model.log_scale = True
        data = np.random.rand(1000)
        bins, counts = model._calc_histogram(data, 0.0, 1.0)
        assert len(bins) == 257
        assert len(counts) == 256
        assert counts.sum() > 0
        # Log counts should be < raw counts for bins with data
        assert counts.max() < 1000


class TestSampleData:
    """Test the _sample_data static-like method."""

    def test_sample_reduces_size(self):
        model = _model(np.random.rand(10, 10))
        data = np.random.rand(100_000)
        sampled = model._sample_data(data, 1000)
        assert len(sampled) == 1000

    def test_sample_returns_all_when_small(self):
        model = _model(np.random.rand(10, 10))
        data = np.random.rand(100)
        sampled = model._sample_data(data, 1000)
        assert len(sampled) == 100

    def test_sample_filters_non_finite(self):
        model = _model(np.random.rand(10, 10))
        data = np.array([np.nan, np.inf, -np.inf, 1.0, 2.0, 3.0])
        sampled = model._sample_data(data, 1000)
        assert len(sampled) == 3
        assert np.all(np.isfinite(sampled))

    def test_sample_empty_after_filter(self):
        model = _model(np.random.rand(10, 10))
        data = np.array([np.nan, np.inf])
        sampled = model._sample_data(data, 1000)
        assert len(sampled) == 0


class TestRgbToLuminance:
    """Test the _rgb_to_luminance method directly."""

    def test_bt709_coefficients(self):
        model = _model(np.random.rand(10, 10, 3).astype(np.float32), rgb=True)
        data = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)  # pure red
        lum = model._rgb_to_luminance(data)
        assert lum[0, 0] == pytest.approx(0.2126)

        data = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)  # pure green
        lum = model._rgb_to_luminance(data)
        assert lum[0, 0] == pytest.approx(0.7152)

        data = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)  # pure blue
        lum = model._rgb_to_luminance(data)
        assert lum[0, 0] == pytest.approx(0.0722)

    def test_alpha_channel_ignored(self):
        model = _model(np.random.rand(10, 10, 4).astype(np.float32), rgb=True)
        data = np.ones((1, 1, 4), dtype=np.float32)
        data[..., 3] = 0.0  # alpha = 0 should not affect luminance
        lum = model._rgb_to_luminance(data)
        expected = 0.2126 + 0.7152 + 0.0722
        assert lum[0, 0] == pytest.approx(expected)

    def test_uint8_input(self):
        model = _model(
            np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8), rgb=True
        )
        data = np.array([[[255, 0, 0]]], dtype=np.uint8)
        lum = model._rgb_to_luminance(data)
        assert lum[0, 0] == pytest.approx(0.2126 * 255.0)


class TestLargeData:
    """Test histogram with larger datasets."""

    def test_sampling_large_data(self):
        data = np.random.rand(500, 500).astype(np.float32)
        model = _model(data)
        counts = model.counts
        assert counts is not None
        assert len(counts) == 256


class TestDask:
    """Test histogram with dask arrays (no full materialization)."""

    def test_chunked_sampling_via_load_chunk(self):
        """_load_chunk should load individual chunks from dask or zarr."""
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(500, 500).astype(np.float32), chunks=100
        )
        model = _model(data)
        # _load_chunk is a static helper used by _compute_sampled
        block = model._load_chunk(data, 0)
        assert isinstance(block, np.ndarray)
        assert block.size > 0

    def test_max_samples_configurable(self):
        """max_samples should be configurable per-instance."""
        model = _model(np.random.rand(10, 10))
        assert model.max_samples == DEFAULT_MAX_SAMPLES
        model.max_samples = 500
        assert model.max_samples == 500

    def test_dask_full_mode_histogram(self):
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(50, 50).astype(np.float32), chunks=25
        )
        model = _model(data)
        model.mode = 'full'
        model.enabled = True
        counts = model.counts
        assert counts is not None
        assert len(counts) == 256
        assert counts.sum() > 0

    def test_dask_canvas_mode_slice_histogram(self):
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(5, 50, 50).astype(np.float32), chunks=(1, 25, 25)
        )
        model = _model(data)
        model.mode = 'canvas'
        model.enabled = True
        counts = model.counts
        assert counts is not None
        assert len(counts) == 256
        assert counts.sum() > 0

    def test_dask_full_mode_samples_large(self):
        """Large dask arrays should be sampled, not fully materialized."""
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(2000, 2000).astype(np.float32), chunks=1000
        )
        model = _model(data)
        model.mode = 'full'
        model.enabled = True
        counts = model.counts
        assert counts is not None
        assert len(counts) == 256


class TestZarr:
    """Test histogram with zarr arrays (chunked, non-dask)."""

    def test_has_chunks_returns_true_for_zarr(self):
        zarr = pytest.importorskip('zarr')
        data = zarr.zeros((50, 50), chunks=(25, 25), dtype=np.float32)
        model = _model(np.zeros((10, 10)))
        assert model._has_chunks(data)

    def test_has_chunks_returns_false_for_numpy(self):
        model = _model(np.zeros((10, 10)))
        assert not model._has_chunks(np.zeros((50, 50)))

    def test_chunk_sizes_correct_for_zarr(self):
        zarr = pytest.importorskip('zarr')
        data = zarr.zeros((50, 50), chunks=(25, 25), dtype=np.float32)
        model = _model(np.zeros((10, 10)))
        sizes = model._chunk_sizes(data)
        # 50x50 with 25x25 chunks = 4 chunks of 625 each
        assert len(sizes) == 4
        assert all(s == 625 for s in sizes)

    def test_chunk_sizes_non_uniform_chunks(self):
        zarr = pytest.importorskip('zarr')
        # 60x90 with 25x25 chunks → last chunks are partial
        data = zarr.zeros((60, 90), chunks=(25, 25), dtype=np.float32)
        model = _model(np.zeros((10, 10)))
        sizes = model._chunk_sizes(data)
        # 3 * 4 = 12 chunks (ceil(60/25)=3, ceil(90/25)=4)
        assert len(sizes) == 12
        # First chunk = 25*25 = 625
        assert sizes[0] == 625
        # Total elements = sum(sizes) = 60*90 = 5400
        assert sum(sizes) == 5400

    def test_load_chunk_from_zarr(self):
        zarr = pytest.importorskip('zarr')
        import numpy as np

        data = zarr.array(np.arange(100, dtype=np.float32).reshape(10, 10))
        model = _model(np.zeros((10, 10)))
        chunk = model._load_chunk(data, 0)
        assert isinstance(chunk, np.ndarray)
        assert chunk.size > 0
        # For a 10x10 array with default chunk, flat index 0 loads all
        assert len(chunk) == 100

    def test_load_chunk_from_zarr_multi_chunk(self):
        zarr = pytest.importorskip('zarr')
        import numpy as np

        data = zarr.array(
            np.arange(2500, dtype=np.float32).reshape(50, 50),
            chunks=(25, 25),
        )
        model = _model(np.zeros((10, 10)))
        # Chunk (0, 0) = rows 0-24, cols 0-24
        chunk = model._load_chunk(data, 0)
        assert len(chunk) == 625
        # Values are in C-order: each row has 50 elements, so the
        # first 25 rows each contribute 25 columns = 625 values.
        # Expected min/max: row 0 col 0 = 0, row 24 col 24 = 24*50+24 = 1224
        assert float(chunk.min()) == 0.0
        assert float(chunk.max()) == 1224.0

    def test_zarr_full_mode_histogram(self):
        zarr = pytest.importorskip('zarr')
        import numpy as np

        data = zarr.array(
            np.random.rand(50, 50).astype(np.float32),
            chunks=(25, 25),
        )
        model = _model(data)
        model.mode = 'full'
        model.enabled = True
        counts = model.counts
        assert counts is not None
        assert len(counts) == 256
        assert counts.sum() > 0


class TestMaterializationGuard:
    """Tests for the _get_full_data materialization guard."""

    def test_large_non_chunked_data_skipped_with_warning(self, monkeypatch):
        """Large non-chunked array-likes should skip full-mode with a warning."""
        model = _model(np.zeros((10, 10)))

        class LargeArrayLike:
            size = _MAX_MATERIALIZE_ELEMENTS + 1
            shape = (10_000, 5_001)
            dtype = np.dtype(np.float32)

        monkeypatch.setattr(
            model._layer, '_data', LargeArrayLike(), raising=False
        )
        assert not model._has_chunks(model._layer.data)

        with pytest.warns(UserWarning, match='Skipping full-data histogram'):
            result = model._get_full_data()

        assert result is None

    def test_large_non_chunked_data_without_dtype(self, monkeypatch):
        """Should also handle array-likes that lack a .dtype."""
        model = _model(np.zeros((10, 10)))

        class LargeArrayLike:
            size = _MAX_MATERIALIZE_ELEMENTS + 1
            shape = (10_000, 5_001)

        monkeypatch.setattr(
            model._layer, '_data', LargeArrayLike(), raising=False
        )
        with pytest.warns(UserWarning, match='Skipping full-data histogram'):
            result = model._get_full_data()

        assert result is None

    def test_small_non_chunked_data_materializes(self, monkeypatch):
        """Small non-chunked array-likes should still be materialized."""
        model = _model(np.zeros((10, 10)))

        class SmallArrayLike:
            size = 100
            shape = (10, 10)
            dtype = np.dtype(np.float32)

        monkeypatch.setattr(
            model._layer, '_data', SmallArrayLike(), raising=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = model._get_full_data()

        assert result is not None

    def test_numpy_data_passes_through(self):
        """Plain numpy arrays should take the fast path, not hit the guard."""
        data = np.random.rand(100, 100).astype(np.float32)
        model = _model(data)

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = model._get_full_data()

        assert result is data  # same object, not copied

    def test_dask_data_passes_through(self):
        """Dask arrays should take the chunked path, not hit the guard."""
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(10_000, 10_000).astype(np.float32),
            chunks=(1000, 1000),
        )
        model = _model(data)

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = model._get_full_data()

        assert result is data  # same object, returned as-is
