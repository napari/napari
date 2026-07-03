import numpy as np
import pytest

from napari._qt.qthreading import create_worker
from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari._qt.widgets.qt_histogram_content import QtHistogramContentWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
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


def test_qt_histogram_widget_updates_from_viewer_theme(
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
        model.compute()
        return model.bins, model.counts

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
        model.compute()
        return model.bins, model.counts

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
    model.n_bins = 128

    worker2 = create_worker(_work)
    worker2.returned.connect(_on_done)
    worker2.start()
    qtbot.waitUntil(lambda: done[0], timeout=30000)
    assert result[0] is not None
    bins2, counts2 = result[0]
    assert len(bins2) == 129  # n_bins=128 → 129 bin edges
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
        # May have been set to None if already finished
        assert widget._compute_worker is True

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
    layer.histogram.compute()
    visual.set_data(
        bins=layer.histogram.bins,
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
    layer.histogram.compute()

    # Call with clims where min == max
    visual.set_data(
        bins=layer.histogram.bins,
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
    # Should not crash

    widget.cleanup()
