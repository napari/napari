import threading

import numpy as np
import pytest

from napari.components.histogram import Histogram
from napari.layers import Image
from napari.utils.histogram import (
    STATE_KEY,
    VALUES_KEY,
    clear_cached_histogram,
    get_cached_histogram,
    histogram_fingerprint,
    set_cached_histogram,
)


@pytest.fixture
def layer():
    return Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))


@pytest.fixture
def values():
    return np.linspace(0, 1, 5), np.ones(4)


def test_cache_miss_then_hit(layer, values):
    edges, counts = values
    assert get_cached_histogram(layer) is None

    set_cached_histogram(layer, edges, counts)

    cached = get_cached_histogram(layer)
    assert cached is not None
    np.testing.assert_allclose(cached[0], edges)
    np.testing.assert_allclose(cached[1], counts)


def test_cache_is_stored_in_layer_metadata(layer, values):
    edges, counts = values
    set_cached_histogram(layer, edges, counts)

    assert VALUES_KEY in layer.metadata
    assert STATE_KEY in layer.metadata
    assert 'canvas' in layer.metadata[VALUES_KEY]
    assert 'canvas' in layer.metadata[STATE_KEY]


def test_state_marker_describes_the_computation(layer, values):
    edges, counts = values
    set_cached_histogram(layer, edges, counts)

    mode = layer.histogram.mode
    state = layer.metadata[STATE_KEY][mode]
    assert state == histogram_fingerprint(layer, mode)
    assert state['mode'] == mode
    assert state['bins'] == layer.histogram.bins
    assert state['log_scale'] == layer.histogram.log_scale
    assert state['contrast_limits_range'] == tuple(layer.contrast_limits_range)


def test_state_marker_reports_staleness(layer, values):
    edges, counts = values
    set_cached_histogram(layer, edges, counts)

    mode = layer.histogram.mode
    state = layer.metadata[STATE_KEY][mode]
    assert state == histogram_fingerprint(layer, mode)

    layer.contrast_limits_range = (0.0, 0.5)
    assert state != histogram_fingerprint(layer, mode)
    assert get_cached_histogram(layer) is None


@pytest.mark.parametrize(
    'change',
    [
        lambda layer: setattr(layer.histogram, 'bins', 32),
        lambda layer: setattr(layer.histogram, 'max_samples', 100),
        lambda layer: setattr(layer.histogram, 'log_scale', True),
        lambda layer: setattr(layer, 'contrast_limits_range', (0.0, 0.5)),
    ],
)
def test_settings_changes_invalidate_the_cache(layer, values, change):
    edges, counts = values
    set_cached_histogram(layer, edges, counts)
    assert get_cached_histogram(layer) is not None

    change(layer)
    assert get_cached_histogram(layer) is None


def test_new_data_invalidates_the_cache(layer, values):
    edges, counts = values
    layer.histogram.mode = 'full'
    set_cached_histogram(layer, edges, counts)
    assert get_cached_histogram(layer) is not None

    layer.data = np.zeros((8, 8), dtype=np.float32)
    assert get_cached_histogram(layer) is None


def test_modes_are_cached_separately(layer, values):
    edges, counts = values

    layer.histogram.mode = 'canvas'
    set_cached_histogram(layer, edges, counts)

    layer.histogram.mode = 'full'
    assert get_cached_histogram(layer) is None
    set_cached_histogram(layer, edges * 2, counts * 2)

    assert set(layer.metadata[VALUES_KEY]) == {'canvas', 'full'}

    layer.histogram.mode = 'canvas'
    cached = get_cached_histogram(layer)
    assert cached is not None
    np.testing.assert_allclose(cached[0], edges)

    layer.histogram.mode = 'full'
    cached = get_cached_histogram(layer)
    assert cached is not None
    np.testing.assert_allclose(cached[0], edges * 2)


def test_layers_have_separate_caches(values):
    edges, counts = values
    layer_a = Image(np.zeros((8, 8), dtype=np.float32))
    layer_b = Image(np.zeros((8, 8), dtype=np.float32))

    set_cached_histogram(layer_a, edges, counts)
    assert get_cached_histogram(layer_b) is None

    set_cached_histogram(layer_b, edges * 3, counts * 3)
    np.testing.assert_allclose(get_cached_histogram(layer_a)[0], edges)
    np.testing.assert_allclose(get_cached_histogram(layer_b)[0], edges * 3)


def test_clear_cached_histogram(layer, values):
    edges, counts = values
    set_cached_histogram(layer, edges, counts)
    clear_cached_histogram(layer)
    assert get_cached_histogram(layer) is None
    assert VALUES_KEY not in layer.metadata
    assert STATE_KEY not in layer.metadata


def test_cache_access_is_threadsafe(layer, values):
    edges, counts = values
    errors: list[Exception] = []

    def hammer():
        try:
            for _ in range(100):
                set_cached_histogram(layer, edges, counts)
                cached = get_cached_histogram(layer)
                if cached is not None:
                    assert len(cached[0]) == len(cached[1]) + 1
        except Exception as err:  # pragma: no cover - only on failure  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert get_cached_histogram(layer) is not None


def test_worker_uses_cached_values(monkeypatch, layer, values, qapp):
    from napari._qt.widgets.qt_histogram import _HistogramWorker

    edges, counts = values
    set_cached_histogram(layer, edges, counts)

    def fail(*args, **kwargs):
        raise AssertionError('the histogram should not be recomputed')

    monkeypatch.setattr(Histogram, '_compute_async_no_events', fail)

    worker = _HistogramWorker(layer)
    result = worker._compute_once(0)
    assert result is not None
    np.testing.assert_allclose(result[0], edges)
    np.testing.assert_allclose(result[1], counts)


def test_worker_populates_the_cache(layer, qapp):
    from napari._qt.widgets.qt_histogram import _HistogramWorker

    worker = _HistogramWorker(layer)
    result = worker._compute_once(0)
    assert result is not None

    cached = get_cached_histogram(layer)
    assert cached is not None
    np.testing.assert_allclose(cached[0], result[0])
    np.testing.assert_allclose(cached[1], result[1])


def test_worker_reuses_the_cache_of_the_previous_mode(
    monkeypatch, layer, qapp
):
    """Each mode keeps its own histogram when switching back and forth."""
    from napari._qt.widgets.qt_histogram import _HistogramWorker

    modes: list[str] = []
    original = Histogram._compute_async_no_events

    def counting(self, layer):
        modes.append(self.mode)
        return original(self, layer)

    monkeypatch.setattr(Histogram, '_compute_async_no_events', counting)

    worker = _HistogramWorker(layer)
    worker._compute_once(0)
    layer.histogram.mode = 'full'
    worker._compute_once(0)
    layer.histogram.mode = 'canvas'
    worker._compute_once(0)

    assert modes == ['canvas', 'full']
