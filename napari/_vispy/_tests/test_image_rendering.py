import numpy as np
import pytest

from napari._tests.utils import skip_on_win_ci
from napari._vispy.layers.image import VispyImageLayer
from napari.layers.image import Image


def test_image_rendering(make_napari_viewer):
    """Test 3D image with different rendering."""
    viewer = make_napari_viewer()

    viewer.dims.ndisplay = 3

    data = np.random.random((20, 20, 20))
    layer = viewer.add_image(data)

    assert layer.rendering == 'mip'

    # Change the interpolation property
    with pytest.deprecated_call():
        layer.interpolation = 'linear'
    assert layer.interpolation2d == 'nearest'
    with pytest.deprecated_call():
        assert layer.interpolation == 'linear'
    assert layer.interpolation3d == 'linear'

    # Change rendering property
    layer.rendering = 'translucent'
    assert layer.rendering == 'translucent'

    # Change rendering property
    layer.rendering = 'attenuated_mip'
    assert layer.rendering == 'attenuated_mip'
    layer.attenuation = 0.15
    assert layer.attenuation == 0.15

    # Change rendering property
    layer.rendering = 'iso'
    assert layer.rendering == 'iso'
    layer.iso_threshold = 0.3
    assert layer.iso_threshold == 0.3

    # Change rendering property
    layer.rendering = 'additive'
    assert layer.rendering == 'additive'


@skip_on_win_ci
def test_visibility_consistency(qapp, make_napari_viewer):
    """Make sure toggling visibility maintains image contrast.

    see #1622 for details.
    """
    viewer = make_napari_viewer(show=True)

    layer = viewer.add_image(
        np.random.random((200, 200)), contrast_limits=[0, 10]
    )
    qapp.processEvents()
    layer.contrast_limits = (0, 2)
    screen1 = viewer.screenshot(flash=False).astype('float')
    layer.visible = True
    screen2 = viewer.screenshot(flash=False).astype('float')
    assert np.max(np.abs(screen2 - screen1)) < 5


def test_clipping_planes_dims():
    """
    Ensure that dims are correctly set on clipping planes
    (vispy uses xyz, napary zyx)
    """
    clipping_planes = {
        'position': (1, 2, 3),
        'normal': (1, 2, 3),
    }
    image_layer = Image(
        np.zeros((2, 2, 2)), experimental_clipping_planes=clipping_planes
    )
    vispy_layer = VispyImageLayer(image_layer)
    napari_clip = image_layer.experimental_clipping_planes.as_array()
    # needed to get volume node
    image_layer._ndisplay = 3
    vispy_layer._on_display_change()
    vispy_clip = vispy_layer.node.clipping_planes
    assert np.all(napari_clip == vispy_clip[..., ::-1])
