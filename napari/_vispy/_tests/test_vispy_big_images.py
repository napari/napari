import numpy as np
import pytest


@pytest.mark.filterwarnings("ignore:data shape:UserWarning")
def test_big_2D_image(make_test_viewer):
    """Test big 2D image with axis exceeding max texture size."""
    viewer = make_test_viewer()

    shape = (20_000, 10)
    data = np.random.random(shape)
    layer = viewer.add_image(data, multiscale=False)
    visual = viewer.window.qt_viewer.layer_to_visual[layer]
    assert visual.node is not None
    if visual.MAX_TEXTURE_SIZE_2D is not None:
        s = np.ceil(np.divide(shape, visual.MAX_TEXTURE_SIZE_2D)).astype(int)
        assert np.all(layer._transforms['tile2data'].scale == s)


@pytest.mark.filterwarnings("ignore:data shape:UserWarning")
def test_big_3D_image(make_test_viewer):
    """Test big 3D image with axis exceeding max texture size."""
    viewer = make_test_viewer(ndisplay=3)

    shape = (5, 10, 3_000)
    data = np.random.random(shape)
    layer = viewer.add_image(data, multiscale=False)
    visual = viewer.window.qt_viewer.layer_to_visual[layer]
    assert visual.node is not None
    if visual.MAX_TEXTURE_SIZE_3D is not None:
        s = np.ceil(np.divide(shape, visual.MAX_TEXTURE_SIZE_3D)).astype(int)
        assert np.all(layer._transforms['tile2data'].scale == s)
