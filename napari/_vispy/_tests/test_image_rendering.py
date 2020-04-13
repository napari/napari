import numpy as np


def test_image_rendering(viewer_factory):
    """Test 3D image with different rendering."""
    view, viewer = viewer_factory()

    data = np.random.random((20, 20, 20))
    layer = viewer.add_image(data)

    assert layer.rendering == 'mip'

    # Change rendering property
    layer.rendering = 'translucent'
    assert layer.rendering == 'translucent'

    # Change rendering property
    layer.rendering = 'attenuated_mip'
    assert layer.rendering == 'attenuated_mip'
    layer.attenuation = 0.2
    assert layer.attenuation == 0.2

    # Change rendering property
    layer.rendering = 'iso'
    assert layer.rendering == 'iso'
    layer.iso_threshold = 0.3
    assert layer.iso_threshold == 0.3

    # Change rendering property
    layer.rendering = 'additive'
    assert layer.rendering == 'additive'
