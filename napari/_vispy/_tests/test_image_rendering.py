import os
import sys

import numpy as np
import pytest


def test_image_rendering(make_napari_viewer):
    """Test 3D image with different rendering."""
    viewer = make_napari_viewer()

    viewer.dims.ndisplay = 3

    data = np.random.random((20, 20, 20))
    layer = viewer.add_image(data)

    assert layer.rendering == 'mip'

    # Change the interpolation property
    layer.interpolation = 'linear'
    assert layer.interpolation == 'linear'

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


@pytest.mark.skipif(
    sys.platform.startswith('win') or not os.getenv("CI"),
    reason='Screenshot tests are not supported on napari windows CI.',
)
def test_visibility_consistency(qtbot, make_napari_viewer):
    """Make sure toggling visibility maintains image contrast.

    see #1622 for details.
    """
    viewer = make_napari_viewer(show=True)

    layer = viewer.add_image(
        np.random.random((200, 200)), contrast_limits=[0, 10]
    )
    qtbot.wait(10)
    layer.contrast_limits = (0, 2)
    screen1 = viewer.screenshot().astype('float')
    layer.visible = True
    screen2 = viewer.screenshot().astype('float')
    assert np.max(np.abs(screen2 - screen1)) < 5
