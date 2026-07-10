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
        # _bin_edges has at least 2 elements (default [0.0, 1.0])
        assert len(model._bin_edges) >= 2
        assert len(model.counts) >= 1

    def test_default_mode_canvas(self):
        model = _model(np.random.rand(10, 10))
        assert model.mode == 'canvas'

    def test_default_bins_256(self):
        model = _model(np.random.rand(10, 10))
        assert model.bins == 256

    def test_default_log_scale_false(self):
        model = _model(np.random.rand(10, 10))
        assert not model.log_scale

    def test_default_max_samples(self):
        model = _model(np.random.rand(10, 10))
        assert model.max_samples == DEFAULT_MAX_SAMPLES


class TestEnableDisable:
    """Test enabling and disabling histogram computation."""

    def test_enable_triggers_computation_for_non_chunked(self):
        """Enabling should trigger compute for non-chunked data."""
        model = _model(np.random.rand(10, 10).astype(np.float32))
        assert not model._layer_events_connected
        model.enabled = True
        assert model._layer_events_connected
        # Non-chunked data computes synchronously on enable.
        assert model._dirty is False
        assert len(model.counts) == 256

    def test_disable_disconnects_layer_events(self):
        """Disabling should disconnect layer events and preserve cached data."""
        model = _model(np.random.rand(10, 10))
        model.enabled = True
        _ = model.counts  # Compute once, clears dirty
        assert model._layer_events_connected

        model.enabled = False
        # Layer events should be disconnected
        assert not model._layer_events_connected
        # Cached bin edges should be preserved (not cleared)
        assert len(model._bin_edges) == 257

    def test_no_computation_when_disabled(self):
        """When disabled, changing layer data should not trigger computation.

        Layer events are disconnected when ``enabled`` is False, so
        ``_mark_dirty`` is never called.  The model should remain in
        its default state until explicitly enabled.
        """
        model = _model(np.random.rand(10, 10))
        model.enabled = False
        # Model starts dirty and enabled=False keeps it that way
        assert model._dirty
        assert list(model._bin_edges) == [0.0, 1.0]
        assert list(model._counts) == [0.0]

        # Change data while disabled — model should stay dirty (no compute)
        # and cached values should remain default.
        model._layer.data = np.random.rand(10, 10)
        assert model._dirty
        assert list(model._bin_edges) == [0.0, 1.0]
        assert list(model._counts) == [0.0]


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
        assert len(counts) == model.bins
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
        """All same value — histogram should handle gracefully.

        For integer constant data, ``_calc_histogram`` expands the range
        by ±0.5, so bin edges span half-integer boundaries.
        """
        model = _model(np.full((10, 10), 5, dtype=np.uint8))
        model.enabled = True
        counts = model.counts
        assert counts.sum() > 0
        # Bin edges should span the expanded range around the constant value
        assert model._bin_edges[0] < 5.0
        assert model._bin_edges[-1] > 5.0
        # All 100 elements should fall into a single bin
        assert counts.sum() == 100
        # At least one bin captures all the data
        assert np.any(counts == 100)

    def test_single_element(self):
        model = _model(np.array([[42.0]], dtype=np.float32))
        model.enabled = True
        counts = model.counts
        assert len(counts) > 0

    def test_with_nan(self):
        """NaN values should be filtered out — all counts should be finite."""
        data = np.random.rand(20, 20)
        data[0, 0] = np.nan
        model = _model(data.astype(np.float32))
        model.enabled = True
        counts = model.counts
        assert counts.sum() > 0
        assert np.all(np.isfinite(counts)), (
            'NaN in data should not produce NaN in counts'
        )

    def test_with_inf(self):
        """Inf values should be filtered out — all counts should be finite."""
        data = np.random.rand(20, 20)
        data[0, 0] = np.inf
        data[0, 1] = -np.inf
        model = _model(data.astype(np.float32))
        model.enabled = True
        counts = model.counts
        assert counts.sum() > 0
        assert np.all(np.isfinite(counts)), (
            'Inf in data should not produce NaN/Inf in counts'
        )


class TestCustomBins:
    """Test custom bin counts."""

    def test_custom_bins(self):
        model = _model(np.random.rand(20, 20), colormap='gray')
        model.enabled = True
        model.bins = 128
        counts = model.counts
        assert len(counts) == 128

    def test_large_bins(self):
        model = _model(np.random.rand(20, 20))
        model.enabled = True
        model.bins = 4096
        _ = model.counts  # Should not crash

    def test_small_bins(self):
        model = _model(np.random.rand(20, 20))
        model.enabled = True
        model.bins = 2
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

    def test_full_result_cached_across_mode_switch(self):
        """full -> canvas -> full restores the cached result, no recompute."""
        model = _model(np.random.rand(5, 20, 20))
        model.enabled = True

        model.mode = 'full'
        full_counts = model.counts.copy()
        gen = model._compute_generation

        model.mode = 'canvas'
        _ = model.counts  # canvas recomputes

        model.mode = 'full'  # should restore from cache, not recompute
        assert not model._dirty
        np.testing.assert_array_equal(model.counts, full_counts)
        # only the canvas switch bumped the generation; the switch back to
        # full restored from cache without re-entering compute()
        assert model._compute_generation == gen + 1

    def test_full_cache_invalidated_by_bins_change(self):
        """A parameter change while away from full forces a recompute."""
        model = _model(np.random.rand(5, 20, 20))
        model.enabled = True
        model.mode = 'full'
        _ = model.counts
        assert model._full_cache is not None

        model.mode = 'canvas'
        model.bins = 64
        assert model._full_cache is None

        model.mode = 'full'
        assert len(model.counts) == 64


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

    def test_log_scale_toggle_triggers_counts_event(self):
        """Toggling log_scale should fire ``events.counts``, not recompute."""
        model = _model(np.random.rand(20, 20))
        model.enabled = True
        _ = model.counts  # initial compute, clears dirty

        events = []
        model.events.counts.connect(lambda: events.append('counts'))
        model.events.bins.connect(lambda: events.append('bins'))

        # Clear the flag set during initial compute
        events.clear()
        model._dirty = False
        model.log_scale = True

        assert 'counts' in events
        assert 'bins' not in events  # no recompute = no bins event

    def test_log_scale_toggle_preserves_bin_edges(self):
        """Bin edges should remain unchanged after log_scale toggle."""
        model = _model(np.random.rand(20, 20))
        model.enabled = True
        _ = model.counts
        original_edges = model._bin_edges.copy()

        model._dirty = False
        model.log_scale = True
        assert np.array_equal(model._bin_edges, original_edges)

        model.log_scale = False
        assert np.array_equal(model._bin_edges, original_edges)


class TestBinsChange:
    """Test changing bins (number of bins)."""

    def test_bins_change_output_length(self):
        model = _model(np.random.rand(20, 20))
        assert len(model.counts) == 256
        model.bins = 128
        assert len(model.counts) == 128
        # bins+1 = 129 bin edges
        assert len(model._bin_edges) == 129

    def test_contrast_limits_range_change(self):
        model = _model(np.random.rand(20, 20))
        model.enabled = True
        model._layer.contrast_limits_range = [0.0, 0.5]
        _ = model.counts  # trigger recompute
        assert model._bin_edges[-1] <= 0.5


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

    def test_multiscale_dask_rgb_full_mode(self):
        """Full-mode histogram on multiscale dask RGB must not raise.

        Regression test: ``_sample_rgb_and_luminance`` previously used
        multi-axis ("nd") fancy indexing, which dask does not support and
        which raised ``NotImplementedError`` ("Don't yet support nd fancy
        indexing").  Because ``NotImplementedError`` subclasses
        ``RuntimeError``, the async worker also swallowed it without firing
        ``finished``, leaving the progress spinner running.  The coarsest
        level here has more pixels than ``max_samples`` so the sampling path
        (the one that used nd fancy indexing) is exercised.
        """
        dask = pytest.importorskip('dask.array')
        levels = [
            dask.from_array(
                np.random.randint(0, 256, (h, h, 3), dtype=np.uint8),
                chunks=(h // 2, h // 2, 3),
            )
            for h in (256, 128, 64)
        ]
        model = _model(levels, rgb=True, multiscale=True)
        model.max_samples = (
            1000  # < coarsest n_pixels (64*64) to force sampling
        )
        model.mode = 'full'
        model.enabled = True
        counts = model.counts
        assert len(counts) == 256
        assert counts.sum() > 0

    def test_sample_rgb_and_luminance_dask(self):
        """_sample_rgb_and_luminance returns a materialized 1D array for dask.

        Exercises the sampling branch (n_pixels > max_samples) directly on a
        dask array to guard the single-axis fancy-indexing implementation.
        """
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
            chunks=(32, 32, 3),
        )
        model = _model(data, rgb=True)
        model.max_samples = 500
        lum = model._sample_rgb_and_luminance(data)
        assert isinstance(lum, np.ndarray)
        assert lum.ndim == 1
        assert lum.size <= 500


class TestEvents:
    """Test histogram event emissions."""

    def test_compute_generator_sets_model_state(self):
        """Calling ``counts`` triggers lazy compute and sets model state."""
        model = _model(np.random.rand(20, 20))
        _ = model.counts  # triggers lazy compute via the generator
        assert len(model._bin_edges) == 257
        assert len(model._counts) == 256
        assert model._counts.sum() > 0


class TestReset:
    """Test the reset method."""

    def test_reset_clears_and_disables(self):
        model = _model(np.random.rand(10, 10))
        model.enabled = True
        _ = model.counts  # trigger compute
        model.reset()
        assert not model.enabled
        assert model.mode == 'canvas'
        assert model.bins == 256
        assert not model.log_scale

    def test_reset_then_re_enable_computes(self):
        """Reset followed by re-enable should produce fresh counts."""
        model = _model(np.random.rand(10, 10))
        model.enabled = True
        _ = model.counts  # trigger initial compute
        model.reset()
        model.enabled = True
        # Non-chunked data computes synchronously on enable.
        assert model._dirty is False
        counts = model.counts
        assert len(counts) == 256

    def test_reset_invalidates_in_flight_compute(self):
        """A compute generator in flight when reset() is called must not
        write its result back over the reset state.

        reset() bumps _compute_generation, so the generator's stale-guard
        trips and it discards its results; it also clears _computing so the
        model is not left stuck (the now-stale generator's generation-gated
        finally will not clear it).
        """
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(100, 100).astype(np.float32), chunks=(50, 50)
        )
        model = _model(np.zeros((10, 10)))
        model._layer = Image(data)
        model.mode = 'full'
        model.enabled = True

        # Start the progressive compute and load one chunk, then pause it —
        # this stands in for a background worker suspended between chunks.
        gen = model.compute()
        next(gen)
        assert model._computing

        model.reset()

        # Draining the stale generator must not overwrite the reset state.
        for _ in gen:
            pass
        assert model._dirty
        np.testing.assert_array_equal(model._counts, np.array([0.0]))
        np.testing.assert_array_equal(model._bin_edges, np.array([0.0, 1.0]))
        # The model is usable again, not stuck with _computing left True.
        assert model._computing is False


class TestCalcHistogram:
    """Test the pure numpy _calc_histogram method directly."""

    def test_basic_histogram(self):
        model = _model(np.random.rand(10, 10))
        data = np.random.rand(1000)
        bins, counts = model._calc_histogram(data, 0.0, 1.0)
        assert len(bins) == 257  # bins + 1
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

    def test_large_data_sampling_canvas_mode(self):
        """Large data should sample in canvas mode.

        Previously sampling only applied in ``mode='full'``, causing
        blocking on large 2D numpy arrays at default zoom.
        Verifies that ``_sample_data`` is called when
        ``data.size > max_samples`` regardless of mode.
        """
        rng = np.random.default_rng(42)
        data = rng.random((2000, 2000)).astype(np.float32)  # 4M > 1M default
        model = _model(data)
        model.enabled = True
        model.max_samples = 100_000
        list(model.compute())
        assert len(model.counts) == 256
        assert model.counts.sum() > 0

    def test_large_data_full_mode_samples(self):
        """Large data in full mode should sample."""
        rng = np.random.default_rng(42)
        data = rng.random((2000, 2000)).astype(np.float32)
        model = _model(data)
        model.mode = 'full'
        model.enabled = True
        model.max_samples = 100_000
        list(model.compute())
        assert len(model.counts) == 256
        assert model.counts.sum() > 0


class TestDask:
    """Test histogram with dask arrays (no full materialization)."""

    def test_chunked_sampling_via_load_chunk(self):
        """_load_chunk should load individual chunks from dask or zarr."""
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(500, 500).astype(np.float32), chunks=100
        )
        model = _model(data)
        # _load_chunk is a static helper used by _compute_chunked_progressive
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


class TestNoneDataPath:
    """Tests for compute() with None/inaccessible data."""

    def test_compute_with_none_data(self, monkeypatch):
        """When _get_data() returns None, compute should produce empty bins/counts."""
        model = _model(np.random.rand(10, 10))
        model.enabled = True
        _ = model.counts  # ensure initial compute succeeds
        assert len(model.counts) == 256

        # Force _get_data to return None
        def _fake_get_data():
            return None

        monkeypatch.setattr(model, '_get_data', _fake_get_data)
        list(model.compute())
        assert len(model._bin_edges) == 2
        assert len(model.counts) == 1

    def test_compute_with_empty_data(self, monkeypatch):
        """When _get_data() returns empty array, should produce empty bins/counts."""
        model = _model(np.random.rand(10, 10))
        model.enabled = True
        _ = model.counts

        def _fake_get_data():
            return np.array([])

        monkeypatch.setattr(model, '_get_data', _fake_get_data)
        list(model.compute())
        assert len(model._bin_edges) == 2
        assert len(model.counts) == 1

    def test_reentrancy_guard(self):
        """Calling compute() while already computing should be a no-op."""

        model = _model(np.random.rand(10, 10))
        model._computing = True
        # Should return immediately without executing
        gen = model.compute()
        with pytest.raises(StopIteration):
            next(gen)
        assert model._computing

    def test_compute_progressive_reentrancy_guard(self):
        """Calling compute_progressive() while already computing should be a no-op."""
        model = _model(np.random.rand(10, 10))
        model._computing = True
        # Should return immediately without yielding
        results = list(model.compute())
        assert len(results) == 0
        assert model._computing

    def test_get_data_none_in_canvas_mode_without_slice(self, monkeypatch):
        """In canvas mode with no slice available, _get_data should return None."""
        model = _model(np.random.rand(10, 10))

        # Monkeypatch _get_slice_raw_data to return None,
        # simulating the state before any slice has been computed
        monkeypatch.setattr(model, '_get_slice_raw_data', lambda: None)

        result = model._get_data()
        assert result is None


class TestEventHandlers:
    """Test histogram event handler wiring."""

    def test_on_slice_change_only_in_canvas_mode(self):
        """_on_slice_change should only trigger recompute in canvas mode."""
        model = _model(np.random.rand(5, 20, 20))
        list(model.compute())  # initial compute, clears dirty

        # In full mode, slice changes are ignored (no access to canvas data).
        model.mode = 'full'
        list(model.compute())  # compute full-mode, clears dirty
        model._on_slice_change()
        assert not model._dirty, (
            'slice change in full mode should not mark dirty'
        )

        # In canvas mode, slice changes mark dirty.
        model.mode = 'canvas'
        list(model.compute())  # compute canvas, clears dirty
        model._on_slice_change()
        assert model._dirty, 'slice change in canvas mode should mark dirty'

    def test_on_enabled_change_connects_events_when_enabled(self):
        """_on_enabled_change should connect layer events when enabled."""
        model = _model(np.random.rand(10, 10))
        model.enabled = True
        _ = model.counts  # initial compute
        model.enabled = False  # disconnect
        assert not model._layer_events_connected

        model._on_enabled_change()  # enabled is False, so disconnect
        assert not model._layer_events_connected

        model.enabled = True
        # enabled=True via _on_enabled_change connects layer events
        assert model._layer_events_connected

    def test_on_enabled_change_skips_when_disabled(self):
        """When enabled is False, _on_enabled_change should not compute."""
        model = _model(np.random.rand(10, 10))
        model._dirty = True
        model.enabled = False
        model._on_enabled_change()
        assert model._dirty  # still dirty, no compute triggered

    def test_on_params_change_triggers_recompute(self):
        """Changing bins or mode should mark dirty and trigger recompute."""
        model = _model(np.random.rand(10, 10))
        model.enabled = True
        list(model.compute())  # initial compute

        model.bins = 128
        # Setting bins fires event → _on_params_change → _mark_dirty
        # → _dirty=True. Iterate compute() to get fresh results.
        list(model.compute())
        assert len(model.counts) == 128

    def test_log_scale_change_transforms_counts_in_place(self):
        """Changing log_scale should transform counts in-place without recompute.

        ``_on_log_scale_change`` applies ``log10(counts + 1)`` to
        existing counts — it does NOT recompute the histogram from
        data.  Verify the transform preserves length and dtype.
        """
        model = _model(np.random.rand(10, 10))
        model.enabled = True
        _ = model.counts  # initial compute

        model.log_scale = True
        counts = model.counts
        assert len(counts) == 256
        assert counts.dtype == np.float32
        # Log-scaled values should be smaller than original raw counts
        assert counts.max() < 256


class TestCalcHistogramExtended:
    """Extended tests for _calc_histogram edge cases."""

    def test_constant_range_expands_float_zero(self):
        """When min==max at zero on float data, delta floor of 0.5 should apply.

        The expansion logic is ``max(0.5, abs(value) * 0.01)`` with a
        fallback of 0.5 when value is 0, so zero-valued constant data
        gets a [-0.5, 0.5] range.
        """
        model = _model(np.array([[0.0]], dtype=np.float32))
        data = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        bins, counts = model._calc_histogram(data, 0.0, 0.0)
        # Range should be [-0.5, 0.5]
        assert bins[0] == -0.5
        assert bins[-1] == 0.5
        assert counts.sum() == 3

    def test_sample_data_returns_empty(self):
        """_sample_data should return an empty array when all values are non-finite."""
        model = _model(np.random.rand(10, 10))
        data = np.array([np.nan, np.inf, -np.inf])
        sampled = model._sample_data(data, 1000)
        assert len(sampled) == 0

    def test_finalize_histogram_falls_back_to_data_range(self, monkeypatch):
        """When contrast_limits_range is None/None, _finalize_histogram
        should fall back to nanmin/nanmax of the data."""
        model = _model(np.random.rand(10, 10).astype(np.float32))
        data = np.random.rand(100).astype(np.float32)
        # Bypass the setter validation by targeting the private backing field
        monkeypatch.setattr(
            model._layer, '_contrast_limits_range', (None, None)
        )
        model._finalize_histogram(data)
        assert len(model._bin_edges) == 257
        assert model._bin_edges[-1] > model._bin_edges[0]

    def test_log_scale_with_chunked_compute(self, monkeypatch):
        """Log scale should be correctly applied in the _compute_chunked_progressive path."""
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(100, 100).astype(np.float32), chunks=(50, 50)
        )
        model = _model(np.zeros((10, 10)))
        # We need to set up the model to go through _compute_chunked_progressive
        model._layer = Image(data)
        model.mode = 'full'
        model.log_scale = True
        model.enabled = True

        counts = model.counts
        assert len(counts) == 256
        # Log-scaled counts should be non-negative
        assert np.all(counts >= 0)


class TestComputeProgressive:
    """Test the compute_progressive generator method."""

    def test_progressive_yields_intermediate_on_chunked_data(self):
        """compute_progressive should yield intermediate results for chunked data."""
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(500, 500).astype(np.float32), chunks=(50, 50)
        )
        model = _model(np.zeros((10, 10)))
        model._layer = Image(data)
        model.mode = 'full'
        model.enabled = True

        # Collect all yielded results
        results = list(model.compute())

        # Should have yielded at least one intermediate result
        assert len(results) >= 1
        for bin_edges, counts in results:
            assert len(bin_edges) == 257  # bins + 1
            assert len(counts) == 256
            assert counts.sum() > 0
            assert bin_edges.dtype == np.float32
            assert counts.dtype == np.float32

        # After completion, model state should be consistent
        assert not model._dirty
        assert len(model._bin_edges) == 257

    def test_progressive_non_chunked_yields_once(self):
        """compute_progressive on non-chunked data should yield the final result once."""
        model = _model(np.random.rand(10, 10))
        model.enabled = True
        results = list(model.compute())
        assert len(results) == 1
        bin_edges, counts = results[0]
        assert len(bin_edges) == 257
        assert len(counts) == 256
        assert counts.sum() > 0

    def test_progressive_handles_none_data(self, monkeypatch):
        """compute_progressive with no data should yield nothing but clear dirty."""
        model = _model(np.random.rand(10, 10))
        model.enabled = True

        def _fake_get_data():
            return None

        monkeypatch.setattr(model, '_get_data', _fake_get_data)
        results = list(model.compute())
        assert len(results) == 0
        assert not model._dirty

    def test_stale_generator_does_not_clear_computing(self):
        """A superseded generator's finally must not clear _computing.

        When a compute is aborted and replaced (e.g. a parameter change
        starts a fresh worker), the old generator can resume later on its
        worker thread and run its finally *after* the replacement set
        _computing=True.  compute()'s finally is generation-gated so the
        stale generator leaves the flag owned by the current generation;
        without the gate it would clear it, defeating the re-entrancy guard.
        """
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(100, 100).astype(np.float32), chunks=(50, 50)
        )
        model = _model(np.zeros((10, 10)))
        model._layer = Image(data)
        model.mode = 'full'
        model.enabled = True

        gen = model.compute()
        next(gen)  # in flight: _computing=True at this generation
        assert model._computing

        # Simulate a replacement compute taking over: it owns _computing and
        # advances the generation past the paused generator's.
        model._compute_generation += 1

        # Draining the now-stale generator must leave _computing set for the
        # replacement rather than clearing it.
        for _ in gen:
            pass
        assert model._computing
