import numpy as np
from napari import view


def test_view_function(qtbot):
    """Test creating viewer with image data."""
    np.random.seed(0)
    data = np.random.random((10, 15))

    viewer = view(data)
    qtbot.addWidget(viewer.window.qt_viewer)

    assert viewer.title == 'napari'
    assert viewer.window.qt_viewer.viewer == viewer

    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)
    assert viewer.dims.ndim == 2

    # Close the viewer
    viewer.window.close()


def test_multiple_images(qtbot):
    """Test creating viewer with mutliple images."""
    np.random.seed(0)
    data_a = np.random.random((10, 15))
    data_b = np.random.random((20, 15))
    data_c = np.random.random((10, 25))

    viewer = view(data_a=data_a, data_b=data_b, data_c=data_c)
    qtbot.addWidget(viewer.window.qt_viewer)

    assert len(viewer.layers) == 3
    assert np.all(viewer.layers['data_a'].data == data_a)
    assert np.all(viewer.layers['data_b'].data == data_b)
    assert np.all(viewer.layers['data_c'].data == data_c)

    # Close the viewer
    viewer.window.close()
