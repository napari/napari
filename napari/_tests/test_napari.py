import numpy as np
import pytest
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
