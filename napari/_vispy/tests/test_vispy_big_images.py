import numpy as np
from napari import Viewer


def test_big_2D_image(qtbot):
    """Test big 2D image with axis exceeding max texture size."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    shape = (20_000, 10)
    data = np.random.random(shape)
    layer = viewer.add_image(data, is_pyramid=False)
    visual = view.layer_to_visual[layer]
    assert visual.node is not None
    if visual.MAX_TEXTURE_SIZE_2D is not None:
        ds = np.ceil(np.divide(shape, visual.MAX_TEXTURE_SIZE_2D)).astype(int)
        assert np.all(layer._scale_view == ds)

    # Close the viewer
    viewer.window.close()


def test_big_3D_image(qtbot):
    """Test big 3D image with axis exceeding max texture size."""
    viewer = Viewer(ndisplay=3)
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    shape = (5, 10, 3_000)
    data = np.random.random(shape)
    layer = viewer.add_image(data, is_pyramid=False)
    visual = view.layer_to_visual[layer]
    assert visual.node is not None
    if visual.MAX_TEXTURE_SIZE_3D is not None:
        ds = np.ceil(np.divide(shape, visual.MAX_TEXTURE_SIZE_3D)).astype(int)
        assert np.all(layer._scale_view == ds)

    # Close the viewer
    viewer.window.close()
