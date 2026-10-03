"""Tests for HistogramModel."""

from __future__ import annotations

import numpy as np
import pytest

from napari.layers import Image
from napari.utils.histogram import (
    _MAX_MATERIALIZE_ELEMENTS,
    _get_computed,
)

_DEFAULT_MAX_SAMPLES = 1_000_000


def _image(data, **kwargs):
    """Create an Image layer (sliced at construction time)."""
    return Image(data, **kwargs)


class TestDefaultState:
    """Test the default state of HistogramModel."""

    def test_default_mode_canvas(self):
        model = _image(np.random.rand(10, 10)).histogram
        assert model.mode == 'canvas'

    def test_default_bins_256(self):
        model = _image(np.random.rand(10, 10)).histogram
        assert model.bins == 256

    def test_default_log_scale_false(self):
        model = _image(np.random.rand(10, 10)).histogram
        assert not model.log_scale

    def test_default_max_samples(self):
        model = _image(np.random.rand(10, 10)).histogram
        assert model.max_samples == _DEFAULT_MAX_SAMPLES


class TestCompute:
    """Test synchronous compute and metadata writing."""

    def test_compute_returns_bin_edges_counts(self):
        layer = _image(np.random.rand(20, 20))
        model = layer.histogram
        bin_edges, counts = model.compute(layer)
        assert len(counts) == model.bins
        assert len(bin_edges) == model.bins + 1

    def test_compute_writes_metadata(self):
        layer = _image(np.random.rand(20, 20))
        model = layer.histogram
        model.compute(layer)
        computed = _get_computed(layer)
        assert len(computed['counts']) == model.bins
        assert len(computed['bin_edges']) == model.bins + 1

    def test_get_computed_default_empty(self):
        layer = _image(np.random.rand(10, 10))
        computed = _get_computed(layer)
        assert len(computed['bin_edges']) == 2
        assert len(computed['counts']) == 1

    def test_compute_emits_events(self):
        layer = _image(np.random.rand(20, 20))
        model = layer.histogram
        events = []
        model.events.updated.connect(lambda: events.append('updated'))
        model.events.completed.connect(lambda: events.append('completed'))
        model.compute(layer)
        assert 'updated' in events
        assert 'completed' in events

    def test_compute_async_writes_metadata_and_emits(self):
        layer = _image(np.random.rand(20, 20))
        model = layer.histogram
        fired = []
        model.events.updated.connect(lambda: fired.append('u'))
        model.events.completed.connect(lambda: fired.append('c'))
        results = list(model.compute_async(layer))
        assert len(results) == 1
        bin_edges, counts = results[0]
        assert np.array_equal(_get_computed(layer)['bin_edges'], bin_edges)
        assert np.array_equal(_get_computed(layer)['counts'], counts)
        assert 'u' in fired
        assert 'c' in fired

    def test_compute_async_no_events_does_not_emit(self):
        layer = _image(np.random.rand(20, 20))
        model = layer.histogram
        fired = []
        model.events.updated.connect(lambda: fired.append('u'))
        model.events.completed.connect(lambda: fired.append('c'))
        list(model._compute_async_no_events(layer))
        # No events fire from the no-events variant...
        assert fired == []
        # ...but it still writes the result into metadata.
        assert 'counts' in layer.metadata['_computed_histogram']


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
        layer = _image(data)
        _bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) == 256
        assert counts.sum() > 0

    def test_all_zeros(self):
        layer = _image(np.zeros((10, 10), dtype=np.float32))
        _, counts = layer.histogram.compute(layer)
        assert len(counts) > 0

    def test_constant_data(self):
        layer = _image(np.full((10, 10), 42.0, dtype=np.float32))
        _, counts = layer.histogram.compute(layer)
        assert len(counts) > 0

    def test_with_nan(self):
        data = np.random.rand(20, 20)
        data[0, 0] = np.nan
        layer = _image(data.astype(np.float32))
        _, counts = layer.histogram.compute(layer)
        assert counts.sum() > 0
        assert np.all(np.isfinite(counts))

    def test_with_inf(self):
        data = np.random.rand(20, 20)
        data[0, 0] = np.inf
        data[0, 1] = -np.inf
        layer = _image(data.astype(np.float32))
        _, counts = layer.histogram.compute(layer)
        assert counts.sum() > 0
        assert np.all(np.isfinite(counts))


class TestCustomBins:
    """Test custom bin counts."""

    def test_custom_bins(self):
        layer = _image(np.random.rand(20, 20))
        layer.histogram.bins = 128
        bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) == 128
        assert len(bin_edges) == 129

    def test_large_bins(self):
        layer = _image(np.random.rand(20, 20))
        layer.histogram.bins = 4096
        _bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) == 4096

    def test_small_bins(self):
        layer = _image(np.random.rand(20, 20))
        layer.histogram.bins = 2
        _bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) == 2


class TestMode:
    """Test canvas vs full mode."""

    def test_canvas_mode(self):
        layer = _image(np.random.rand(5, 20, 20))
        _bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) > 0

    def test_full_mode(self):
        layer = _image(np.random.rand(5, 20, 20))
        layer.histogram.mode = 'full'
        _bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) > 0

    def test_full_mode_uses_coarsest_multiscale(self):
        data = [
            np.random.rand(100, 100).astype(np.float32),
            np.random.rand(50, 50).astype(np.float32),
        ]
        layer = _image(data, multiscale=True)
        layer.histogram.mode = 'full'
        _, counts = layer.histogram.compute(layer)
        assert counts.sum() == data[-1].size


class TestLogScale:
    """Test log scale."""

    def test_log_scale_differs(self):
        layer = _image(np.random.rand(20, 20))
        model = layer.histogram
        _, linear = model.compute(layer)
        model.log_scale = True
        _, logged = model.compute(layer)
        assert len(logged) == len(linear)
        assert not np.array_equal(linear, logged)

    def test_log_scale_values_smaller(self):
        layer = _image(np.random.rand(20, 20))
        model = layer.histogram
        _, linear = model.compute(layer)
        model.log_scale = True
        _, logged = model.compute(layer)
        assert logged.max() <= linear.max()


class TestRGB:
    """Test histogram with RGB data."""

    def test_rgb_image(self):
        data = np.random.randint(0, 256, size=(20, 20, 3), dtype=np.uint8)
        layer = _image(data, rgb=True)
        _, counts = layer.histogram.compute(layer)
        assert len(counts) > 0


class TestNoneDataPath:
    """Tests for compute() with unavailable data."""

    def test_compute_with_no_display_data(self, monkeypatch):
        """When there is no displayed data, compute yields the empty result."""
        import napari.utils.histogram as hm

        layer = _image(np.random.rand(10, 10))
        monkeypatch.setattr(hm, '_get_data', lambda layer, mode: None)
        bin_edges, counts = layer.histogram.compute(layer)
        assert len(bin_edges) == 2
        assert len(counts) == 1
        assert counts[0] == 0


class TestChunked:
    """Tests for chunked (dask / zarr) data."""

    def test_dask_full_mode_progressive_yields(self):
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(500, 500).astype(np.float32), chunks=(50, 50)
        )
        layer = _image(data)
        layer.histogram.mode = 'full'
        results = list(layer.histogram.compute_async(layer))
        assert len(results) >= 1
        _bin_edges, counts = results[-1]
        assert len(counts) == 256
        assert counts.sum() > 0

    def test_dask_canvas_mode(self):
        dask = pytest.importorskip('dask.array')
        data = dask.from_array(
            np.random.rand(5, 50, 50).astype(np.float32), chunks=(1, 25, 25)
        )
        layer = _image(data)
        _bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) == 256
        assert counts.sum() > 0

    def test_zarr_full_mode(self):
        zarr = pytest.importorskip('zarr')
        data = zarr.array(
            np.random.rand(50, 50).astype(np.float32), chunks=(25, 25)
        )
        layer = _image(data)
        layer.histogram.mode = 'full'
        _bin_edges, counts = layer.histogram.compute(layer)
        assert len(counts) == 256
        assert counts.sum() > 0


class TestMaterializationGuard:
    """Tests for the _get_full_data materialization guard."""

    def test_large_non_chunked_data_skipped_with_warning(self):
        from napari.utils.histogram import _get_full_data

        class Big:
            size = _MAX_MATERIALIZE_ELEMENTS + 1
            shape = (10_000, 5_001)
            dtype = np.dtype(np.float32)

        class FakeLayer:
            data = Big()

        with pytest.warns(UserWarning, match='Skipping full-data histogram'):
            result = _get_full_data(FakeLayer())
        assert result is None
