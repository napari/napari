import numpy as np
import pytest

from napari._qt.qthreading import create_worker
from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari._qt.widgets.qt_histogram_content import QtHistogramContentWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
from napari.components.histogram import HistogramModel
from napari.layers import Image
from napari.settings import get_settings
from napari.utils.theme import get_theme


def test_qt_histogram_settings_mode_sync(qtbot):
    """Settings widget mode combobox should sync bidirectionally with model."""
    layer = Image(np.random.rand(10, 10))
    model = layer.histogram
    widget = QtHistogramSettingsWidget(model)
    qtbot.addWidget(widget)

    # Default state
    assert widget.mode_combobox.currentText() == 'canvas'
    assert model.mode == 'canvas'

    # UI → model: changing combobox updates model
    widget.mode_combobox.setCurrentText('full')
    assert model.mode == 'full'

    # Model → UI: changing model updates combobox
    model.mode = 'canvas'
    assert widget.mode_combobox.currentText() == 'canvas'

    widget.cleanup()


def test_qt_histogram_settings_log_scale_sync(qtbot):
    """Settings widget log scale checkbox should sync bidirectionally with model."""
    layer = Image(np.random.rand(10, 10))
    model = layer.histogram
    widget = QtHistogramSettingsWidget(model)
    qtbot.addWidget(widget)

    # Default state
    assert not widget.log_scale_checkbox.isChecked()
    assert not model.log_scale

    # UI → model: toggling checkbox updates model
    widget.log_scale_checkbox.setChecked(True)
    assert model.log_scale

    # Model → UI: changing model updates checkbox
    model.log_scale = False
    assert not widget.log_scale_checkbox.isChecked()

    widget.cleanup()


def test_qt_histogram_content_composition_and_cleanup(qtbot):
    """Content widget should create histogram + settings children and clean up."""
    layer = Image(np.random.rand(10, 10))
    content = QtHistogramContentWidget(layer)
    qtbot.addWidget(content)

    # Both child widgets exist
    assert content.histogram_widget is not None
    assert content.settings_widget is not None
    assert content.settings_widget.mode_combobox is not None
    assert content.settings_widget.log_scale_checkbox is not None

    # Settings controls are functional
    content.settings_widget.mode_combobox.setCurrentText('full')
    assert layer.histogram.mode == 'full'
    content.settings_widget.log_scale_checkbox.setChecked(True)
    assert layer.histogram.log_scale

    # Cleanup does not crash
    content.cleanup()


def test_qt_histogram_widget_updates_theme(qtbot):
    settings = get_settings()
    old_theme = settings.appearance.theme
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    layer.histogram.enabled = True
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    layer.histogram.compute()

    try:
        settings.appearance.theme = 'light'
        light_theme = get_theme('light')

        qtbot.waitUntil(
            lambda: np.allclose(
                widget.canvas.bgcolor.rgba[:3],
                np.array(light_theme.canvas.as_rgb_tuple()) / 255,
            )
        )

        assert widget.histogram_visual._lut_color == (
            *(
                np.array(light_theme.highlight.as_rgb_tuple(), dtype=float)
                / 255
            ),
            0.95,
        )
        assert widget.histogram_visual._axes_color == (
            *(np.array(light_theme.text.as_rgb_tuple(), dtype=float) / 255),
            0.7,
        )
    finally:
        settings.appearance.theme = old_theme
        widget.cleanup()


def test_qt_histogram_widget_updates_from_settings_theme(
    make_napari_viewer, qtbot
):
    """Histogram widget responds to theme changes via settings (canonical source)."""
    settings = get_settings()
    old_theme = settings.appearance.theme
    viewer = make_napari_viewer()
    layer = viewer.add_image(
        np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    )
    controls = viewer.window._qt_viewer.controls.widgets[layer]
    controls._histogram_control.ensure_content()
    widget = controls._histogram_control.histogram_widget
    assert widget is not None

    qtbot.addWidget(widget)
    layer.histogram.enabled = True
    layer.histogram.compute()

    try:
        settings.appearance.theme = 'light'
        light_theme = get_theme('light')

        qtbot.waitUntil(
            lambda: np.allclose(
                widget.canvas.bgcolor.rgba[:3],
                np.array(light_theme.canvas.as_rgb_tuple()) / 255,
            )
        )

        assert widget.histogram_visual._lut_color == (
            *(
                np.array(light_theme.highlight.as_rgb_tuple(), dtype=float)
                / 255
            ),
            0.95,
        )
    finally:
        settings.appearance.theme = old_theme


def test_qt_histogram_async_compute_with_dask(qtbot):
    """Histogram compute on chunked dask data should work via create_worker."""
    dask = pytest.importorskip('dask.array')
    data = dask.random.random((500, 500), chunks=(50, 50))
    layer = Image(data)
    model = layer.histogram
    model.mode = 'full'
    model.max_samples = 50000

    done = [False]
    result = [None]

    def _work():
        list(model.compute())
        return model._bin_edges, model._counts

    def _on_done(bins_counts):
        result[0] = bins_counts
        done[0] = True

    worker = create_worker(_work)
    worker.returned.connect(_on_done)
    worker.start()

    qtbot.waitUntil(lambda: done[0], timeout=10000)

    assert result[0] is not None
    bins, counts = result[0]
    assert len(bins) == 257
    assert len(counts) == 256
    assert counts.sum() > 0


def test_qt_histogram_sequential_async_with_param_change(qtbot):
    """Sequential async computes with different parameters should each produce correct results.

    This tests the typical pattern where one async compute
    finishes before the next is triggered by a parameter change.
    """
    dask = pytest.importorskip('dask.array')
    data = dask.random.random((500, 500), chunks=(50, 50))
    layer = Image(data)
    model = layer.histogram
    model.mode = 'full'
    model.max_samples = 50000

    done = [False]
    result = [None]

    def _work():
        list(model.compute())
        return model._bin_edges, model._counts

    def _on_done(bins_counts):
        result[0] = bins_counts
        done[0] = True

    # Start first worker and wait for it to complete
    worker1 = create_worker(_work)
    worker1.returned.connect(_on_done)
    worker1.start()
    qtbot.waitUntil(lambda: done[0], timeout=30000)
    assert result[0] is not None
    bins1, counts1 = result[0]
    assert len(bins1) == 257
    assert counts1.sum() > 0

    # Change a parameter and run a second async compute
    done[0] = False
    result[0] = None
    model.bins = 128

    worker2 = create_worker(_work)
    worker2.returned.connect(_on_done)
    worker2.start()
    qtbot.waitUntil(lambda: done[0], timeout=30000)
    assert result[0] is not None
    bins2, counts2 = result[0]
    assert len(bins2) == 129  # bins=128 → 129 bin edges
    assert counts2.sum() > 0


def test_qt_histogram_teardown_during_async_compute(qtbot):
    """Closing a histogram widget while an async compute is in flight
    should not crash.  This guards against the race where a background
    worker's ``finished`` signal fires after ``cleanup()`` has already
    disconnected psygnal events and destroyed the canvas.
    """
    dask = pytest.importorskip('dask.array')
    data = dask.random.random((500, 500), chunks=(50, 50))
    layer = Image(data)
    layer.histogram.mode = 'full'
    layer.histogram.max_samples = 50000

    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)
    layer.histogram.enabled = True

    # Trigger async compute by calling the internal path used in production
    widget._ensure_histogram_computed()
    # Capture the worker before cleanup (cleanup sets _compute_worker to None)
    worker = widget._compute_worker

    # Immediately clean up — simulates viewer close while worker is running
    widget.cleanup()

    # No crash = success

    # Wait for the worker to finish so the thread pool drains before
    # the conftest's _dangling_qthread_pool fixture checks for leaks.
    if worker is not None:
        from qtpy.QtCore import QThreadPool

        qtbot.waitUntil(
            lambda: QThreadPool.globalInstance().activeThreadCount() == 0,
            timeout=10000,
        )


def test_qt_histogram_widget_ensure_computed_worker_cancel(qtbot):
    """Calling _ensure_histogram_computed while a worker is already running
    should cancel the previous worker."""
    dask = pytest.importorskip('dask.array')
    data = dask.random.random((500, 500), chunks=(50, 50))
    layer = Image(data)
    layer.histogram.mode = 'full'
    layer.histogram.max_samples = 50000

    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)
    layer.histogram.enabled = True

    # First call starts a worker
    widget._ensure_histogram_computed()
    first_worker = widget._compute_worker
    assert first_worker is not None

    # Second call should cancel the first and start a new one
    widget._ensure_histogram_computed()
    if widget._compute_worker is not None:
        # If a new worker was created (the first had not yet finished),
        # it should be a different object. If the first already finished,
        # _compute_worker may be None — that's OK too.
        assert widget._compute_worker is not first_worker

    widget.cleanup()

    # Drain thread pool
    from qtpy.QtCore import QThreadPool

    qtbot.waitUntil(
        lambda: QThreadPool.globalInstance().activeThreadCount() == 0,
        timeout=10000,
    )


def test_qt_histogram_widget_ensure_content_disconnect(qtbot):
    """Calling cleanup on QtHistogramContentWidget should disconnect events."""
    layer = Image(np.random.rand(10, 10))
    content = QtHistogramContentWidget(layer)
    qtbot.addWidget(content)

    layer.histogram.enabled = True
    layer.histogram.compute()

    # Before cleanup, changing log_scale triggers recompute
    assert content.histogram_widget._updating is not None  # widget is alive

    content.cleanup()
    # After cleanup, changing log_scale should not crash
    layer.histogram.log_scale = True


def test_histogram_visual_set_data_clear_path(qtbot):
    """Calling set_data with no bins/counts should clear the visual."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual

    # First set some data to get a non-empty state
    layer.histogram.enabled = True
    list(layer.histogram.compute())
    visual.set_data(
        bin_edges=layer.histogram._bin_edges,
        counts=layer.histogram.counts,
        gamma=1.0,
        clims=(0.25, 0.75),
        data_range=(0, 1),
    )

    # Now call set_data with None to trigger _clear path
    visual.set_data()
    # After clear, gamma should be reset to 1.0
    assert visual._gamma == 1.0
    assert visual._clims is None
    assert visual._data_range is None

    widget.cleanup()


def test_histogram_visual_update_lut_line_clims_equal(qtbot):
    """LUT line should handle equal clim values gracefully."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual
    layer.histogram.enabled = True
    list(layer.histogram.compute())

    # Call with clims where min == max
    visual.set_data(
        bin_edges=layer.histogram._bin_edges,
        counts=layer.histogram.counts,
        gamma=1.0,
        clims=(0.5, 0.5),  # equal clims
        data_range=(0, 1),
    )
    # Should not crash; uses the else branch in _update_lut_line
    assert visual._clims == (0.5, 0.5)

    widget.cleanup()


def test_histogram_visual_destroy(qtbot):
    """Calling destroy on the histogram visual should clean up sub-visuals."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual

    # destroy should not crash
    visual.destroy()

    widget.cleanup()


def test_histogram_visual_update_bars_empty(qtbot):
    """_update_bars with fewer than 2 bins should call _set_empty_data."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual

    # Call _update_bars directly with a single bin (len(bins) < 2)
    visual._update_bars(np.array([0.0]), np.array([5.0]))
    # Should not crash; calls _set_empty_data internally
    # After _set_empty_data, the bars mesh should have 3 dummy vertices
    assert visual._bars.mesh_data.get_vertices() is not None

    widget.cleanup()


def test_histogram_visual_update_bars_zero_range(qtbot):
    """_update_bars should handle zero bin range (all bins identical)."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual

    # All bins have the same value → bin_range == 0 → should use bin_range = 1
    bins = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    counts = np.array([10.0, 5.0], dtype=np.float32)
    visual._update_bars(bins, counts)
    # Should not crash; with 2 bins, should produce 8 vertices (4 per bar)
    vertices = visual._bars.mesh_data.get_vertices()
    assert vertices is not None
    assert len(vertices) == 8, (
        'zero-range bars should produce 8 vertices for 2 bins'
    )

    widget.cleanup()


def test_qt_histogram_layer_bar_color(qtbot):
    """_layer_bar_color should return a 4-tuple based on the layer's colormap."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    # Default colormap (gray) → bar color should be a 4-element tuple
    color = widget._layer_bar_color()
    assert len(color) == 4
    assert all(0 <= c <= 1 for c in color)

    # With a reversed colormap, the bar color should still be non-zero
    # (the method uses map([0.8]) to avoid black-on-black for gray_r)
    layer.colormap = 'gray_r'
    color_r = widget._layer_bar_color()
    assert len(color_r) == 4
    # Even on a reversed colormap, the 0.8 position is near-white, so at
    # least one channel should be > 0.5.
    assert any(c > 0.5 for c in color_r), (
        f'gray_r bar color should be light, got {color_r}'
    )

    widget.cleanup()


def test_qt_histogram_mode_switch_uses_async_for_chunked_data(qtbot):
    """Switching to full mode on chunked data should use async compute,
    not block the main thread on synchronous chunk I/O.

    Regression test for the issue where setting mode='full' on a
    dask- or zarr-backed Image layer would synchronously iterate
    chunks in HistogramModel._mark_dirty()/compute(), blocking the
    viewer while each chunk was loaded over I/O (e.g. remote zarr).

    The fix: _mark_dirty() skips compute() for chunked+full data,
    and QtHistogramWidget._on_model_mode_change() triggers the
    GeneratorWorker-based async path instead.
    """
    dask = pytest.importorskip('dask.array')
    data = dask.random.random((500, 500), chunks=(50, 50))
    layer = Image(data)
    layer.histogram.enabled = True

    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    # Initial state: model is clean from the canvas-mode compute that
    # ran during __init__.
    assert not layer.histogram._dirty

    # Switch to full mode.  In the buggy code this would trigger
    # _mark_dirty() → compute() → synchronous chunk iteration.
    layer.histogram.mode = 'full'

    # After _mark_dirty() with the fix: _dirty should be True but
    # compute() should NOT have been called (it was deferred for
    # chunked data).  If _dirty is False here, compute() ran
    # synchronously — the regression.
    assert layer.histogram._dirty, (
        '_mark_dirty() called compute() synchronously on mode switch '
        'with chunked data — this would block the main thread'
    )

    # The widget should have started an async worker via
    # _on_model_mode_change() → _ensure_histogram_computed().
    # For small in-memory dask arrays the worker may already have
    # finished, but if it's still running or just-completed we
    # verify that the async path was taken by waiting for results.
    qtbot.waitUntil(
        lambda: not layer.histogram._dirty,
        timeout=30000,
    )

    # Verify valid histogram results from the async path
    assert len(layer.histogram._bin_edges) == 257
    assert len(layer.histogram.counts) == 256
    assert layer.histogram.counts.sum() > 0

    widget.cleanup()


def _full_data_counts(base, hist, clims_range):
    """Ground-truth full-data histogram for comparison with the model."""
    ground_truth, _ = np.histogram(
        base.ravel(),
        bins=hist.bins,
        range=tuple(float(v) for v in clims_range),
    )
    return ground_truth.astype(np.int64)


def test_two_views_share_single_worker_and_both_animate(qtbot):
    """Two histogram views on one layer (inline + popup) share a single
    compute worker, yet *both* animate progressively — regardless of which
    view owns the worker.

    Regression test for the popup staying blank while the inline animated:
    the two views used to run competing workers over the shared model,
    corrupting the progressive accumulation, and only the worker-owning view
    rendered chunk-by-chunk.
    """
    dask = pytest.importorskip('dask.array')
    # Deterministic data with all chunks sampled (size < max_samples), so the
    # final accumulation is exactly a full-data np.histogram.
    base = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    layer = Image(dask.from_array(base, chunks=(32, 32)))  # 64 chunks
    layer.histogram.mode = 'full'

    # Whichever view reaches _ensure_histogram_computed first owns the single
    # worker; the other must still animate from partial_computed broadcasts.
    view_a = QtHistogramWidget(layer)
    view_b = QtHistogramWidget(layer)
    qtbot.addWidget(view_a)
    qtbot.addWidget(view_b)

    # Count how many times each view's visual is actually redrawn.
    draws = {'a': 0, 'b': 0}
    orig_a = view_a.histogram_visual.set_data
    orig_b = view_b.histogram_visual.set_data

    def count_a(*args, **kwargs):
        draws['a'] += 1
        return orig_a(*args, **kwargs)

    def count_b(*args, **kwargs):
        draws['b'] += 1
        return orig_b(*args, **kwargs)

    view_a.histogram_visual.set_data = count_a
    view_b.histogram_visual.set_data = count_b

    hist = layer.histogram
    layer.histogram.enabled = True  # triggers the shared async compute

    qtbot.waitUntil(
        lambda: not hist._compute_scheduled and not hist._dirty,
        timeout=15000,
    )

    # Both views animated progressively from the single worker's chunks.
    assert draws['a'] > 1
    assert draws['b'] > 1
    # Single-worker invariant held to completion — nothing left dangling.
    assert not hist._compute_scheduled
    assert view_a._compute_worker is None
    assert view_b._compute_worker is None
    # The result is the full accumulation, not a single-chunk fragment.
    assert np.array_equal(
        hist._counts.astype(np.int64),
        _full_data_counts(base, hist, layer.contrast_limits_range),
    )

    view_a.cleanup()
    view_b.cleanup()
    from qtpy.QtCore import QThreadPool

    qtbot.waitUntil(
        lambda: QThreadPool.globalInstance().activeThreadCount() == 0,
        timeout=10000,
    )


def test_closing_owning_view_mid_compute_hands_off_to_survivor(qtbot):
    """Closing the view that owns the in-flight worker nudges a surviving
    view to finish the compute, instead of stranding it with partial data.

    Covers closing the contrast-limits popup mid-load while the inline
    histogram remains open.
    """
    dask = pytest.importorskip('dask.array')
    base = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    layer = Image(dask.from_array(base, chunks=(32, 32)))
    layer.histogram.mode = 'full'

    view_a = QtHistogramWidget(layer)
    view_b = QtHistogramWidget(layer)
    qtbot.addWidget(view_a)
    qtbot.addWidget(view_b)

    hist = layer.histogram
    layer.histogram.enabled = True  # starts the shared worker synchronously

    # Identify the owning view and close it before the compute finishes.
    assert hist._compute_scheduled
    if view_a._compute_worker is not None:
        owner, survivor = view_a, view_b
    else:
        owner, survivor = view_b, view_a
    assert hist._dirty  # compute has not completed yet
    owner.cleanup()

    # The survivor must take over and finish the compute correctly.
    qtbot.waitUntil(
        lambda: not hist._compute_scheduled and not hist._dirty,
        timeout=15000,
    )
    assert survivor._compute_worker is None
    assert np.array_equal(
        hist._counts.astype(np.int64),
        _full_data_counts(base, hist, layer.contrast_limits_range),
    )

    survivor.cleanup()
    from qtpy.QtCore import QThreadPool

    qtbot.waitUntil(
        lambda: QThreadPool.globalInstance().activeThreadCount() == 0,
        timeout=10000,
    )


# pytest-qt would capture the re-raised error as a test failure; opt out.
@pytest.mark.qt_no_exception_capture
def test_persistent_chunk_load_error_does_not_retry_forever(
    qtbot, monkeypatch
):
    """A persistent chunk-load failure in full mode must not spawn an
    unbounded stream of retry workers.

    Regression test: when ``compute()`` raises mid-chunk (e.g. a remote zarr
    read fails), it leaves ``_dirty=True`` because it never reached a clean
    result.  ``_on_async_compute_done`` used to emit ``events.counts()``
    unconditionally on ``finished`` (which fires on error too), re-entering
    ``_on_model_event`` — still dirty + enabled — and spawning a replacement
    worker that repeated the identical failing I/O forever.  The fix skips
    the counts re-emit while the model is still dirty; the worker's
    notification mixin already surfaces the error to the user.

    Each spawned worker reaches exactly one ``_load_chunk`` call before it
    raises, so the ``_load_chunk`` invocation count is the worker count.
    """
    dask = pytest.importorskip('dask.array')
    data = dask.random.random((200, 200), chunks=(50, 50))
    layer = Image(data)
    layer.histogram.mode = 'full'

    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    load_calls = {'n': 0}

    def boom(*args, **kwargs):
        load_calls['n'] += 1
        raise OSError('simulated remote chunk read failure')

    monkeypatch.setattr(HistogramModel, '_load_chunk', staticmethod(boom))

    # Enabling triggers the shared async compute for chunked full-mode data.
    layer.histogram.enabled = True

    # Wait for the single worker to run, fail, and release the compute slot.
    qtbot.waitUntil(
        lambda: (
            not layer.histogram._compute_scheduled
            and widget._compute_worker is None
        ),
        timeout=15000,
    )

    # Give the event loop room to spin: a retry loop would keep spawning
    # workers (and calling _load_chunk) during this window.
    qtbot.wait(300)

    # Exactly one worker ran — the persistent failure was not retried.
    assert load_calls['n'] == 1
    # The model correctly stayed dirty (no valid result was produced) ...
    assert layer.histogram._dirty
    # ... and the compute slot was released, not leaked.
    assert not layer.histogram._compute_scheduled
    assert widget._compute_worker is None

    widget.cleanup()
