import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

import napari
from napari._tests.utils import (
    check_viewer_functioning,
    layer_test_data,
    view_layer_type,
)


@pytest.mark.parametrize('layer_type, data, ndim', layer_test_data)
def test_view(qtbot, layer_type, data, ndim):

    np.random.seed(0)
    viewer = view_layer_type(layer_type, data)
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    check_viewer_functioning(viewer, view, data, ndim)

    # Close the viewer
    viewer.window.close()


def test_view_multichannel(qtbot):
    """Test adding image."""

    np.random.seed(0)
    data = np.random.random((15, 10, 5))
    viewer = napari.view_image(data, channel_axis=-1)
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    assert len(viewer.layers) == data.shape[-1]
    for i in range(data.shape[-1]):
        assert np.all(viewer.layers[i].data == data.take(i, axis=-1))

    # Close the viewer
    viewer.window.close()


def test_widget_cleanup(qtbot):
    """Test that closing the viewer doesn't leave any orphaned widgets."""
    app = QApplication.instance()
    assert len(app.topLevelWidgets()) == 0
    viewer = napari.Viewer()
    assert app.topLevelWidgets()
    viewer.close()
    app.processEvents()
    # unable to get the very last QMainWindow to clean up in pytest...
    # but all the other widgets should be gone
    assert len(app.topLevelWidgets()) == 1
