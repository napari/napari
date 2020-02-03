import sys
import time

import numpy as np
import pytest
from qtpy.QtCore import QEventLoop
from qtpy.QtWidgets import QApplication

from napari import Viewer
from napari._tests.utils import (
    check_viewer_functioning,
    add_layer_by_type,
    layer_test_data,
)


def test_viewer(qtbot):
    """Test instantiating viewer."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    assert viewer.title == 'napari'
    assert view.viewer == viewer

    assert len(viewer.layers) == 0
    assert view.layers.vbox_layout.count() == 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Switch to 3D rendering mode and back to 2D rendering mode
    viewer.dims.ndisplay = 3
    assert viewer.dims.ndisplay == 3
    viewer.dims.ndisplay = 2
    assert viewer.dims.ndisplay == 2

    # Run all class keybindings
    for func in viewer.class_keymap.values():
        func(viewer)
        # the `play` keybinding calls QtDims.play_dim(), which then creates a new QThread.
        # we must then run the keybinding a second time, which will call QtDims.stop(),
        # otherwise the thread will be killed at the end of the test without cleanup,
        # causing a segmentation fault.  (though the tests still pass)
        if func.__name__ == 'play':
            func(viewer)

    if sys.platform == 'darwin':
        # on some versions of Darwin, exiting while fullscreen seems to tickle
        # some bug deep in NSWindow.  This forces the fullscreen keybinding
        # test to complete its draw cycle, then pop back out of fullscreen.
        start = time.time()
        while time.time() < start + 1:
            qapp = QApplication.instance()
            qapp.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 0)

        viewer.window._qt_window.showNormal()

    viewer.window.close()


@pytest.mark.first  # provided by pytest-ordering
def test_no_qt_loop():
    """Test informative error raised when no Qt event loop exists.

    Logically, this test should go at the top of the file. Howveer, that
    resulted in tests passing when only this file was run, but failing when
    other tests involving Qt-bot were run before this file. Putting this test
    second provides a sanity check that pytest-ordering is correctly doing its
    magic.
    """
    with pytest.raises(RuntimeError):
        _ = Viewer()


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
@pytest.mark.parametrize('visible', [True, False])
def test_add_layer(qtbot, layer_class, data, ndim, visible):
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)
    layer = add_layer_by_type(viewer, layer_class, data, visible=visible)
    check_viewer_functioning(viewer, view, data, ndim)

    # Run all class keybindings
    for func in layer.class_keymap.values():
        func(layer)

    # Close the viewer
    viewer.window.close()


def test_screenshot(qtbot):
    "Test taking a screenshot"
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    np.random.seed(0)
    # Add image
    data = np.random.random((10, 15))
    viewer.add_image(data)

    # Add labels
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)

    # Add points
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)

    # Add vectors
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)

    # Add shapes
    data = 20 * np.random.random((10, 4, 2))
    viewer.add_shapes(data)

    # Take screenshot
    screenshot = viewer.screenshot()
    assert screenshot.ndim == 3

    # Take screenshot with the viewer included
    screenshot = viewer.screenshot(with_viewer=True)
    assert screenshot.ndim == 3

    # Close the viewer
    viewer.window.close()


def test_update(qtbot):
    import time

    data = np.random.random((512, 512))
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    layer = viewer.add_image(data)

    def layer_update(*, update_period, num_updates):
        # number of times to update

        for k in range(num_updates):
            time.sleep(update_period)

            dat = np.random.random((512, 512))
            layer.data = dat

            assert layer.data.all() == dat.all()

    viewer.update(layer_update, update_period=0.01, num_updates=100)

    # if we do not sleep, main thread closes before update thread finishes and many qt components get cleaned
    time.sleep(3)

    # Close the viewer
    viewer.window.close()
