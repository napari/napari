from itertools import permutations

import numpy as np
import pytest

from napari._vispy._tests.utils import vispy_image_scene_size
from napari._vispy.layers.image import VispyImageLayer
from napari.components.dims import Dims
from napari.layers import Image


@pytest.mark.parametrize('order', permutations((0, 1, 2)))
def test_3d_slice_of_2d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small square when displayed in 3D.
    """
    image = Image(np.zeros((4, 2)), scale=(1, 2))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(Dims(ndim=3, ndisplay=3, order=order))

    scene_size = vispy_image_scene_size(vispy_image)
    np.testing.assert_array_equal((4, 4, 1), scene_size)


@pytest.mark.parametrize('order', permutations((0, 1, 2)))
def test_2d_slice_of_3d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small square when displayed in 2D.
    """
    image = Image(np.zeros((8, 4, 2)), scale=(1, 2, 4))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(Dims(ndim=3, ndisplay=2, order=order))

    scene_size = vispy_image_scene_size(vispy_image)
    np.testing.assert_array_equal((8, 8, 0), scene_size)


@pytest.mark.parametrize('order', permutations((0, 1, 2)))
def test_3d_slice_of_3d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small cube when displayed in 3D.
    """
    image = Image(np.zeros((8, 4, 2)), scale=(1, 2, 4))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(Dims(ndim=3, ndisplay=3, order=order))

    scene_size = vispy_image_scene_size(vispy_image)
    np.testing.assert_array_equal((8, 8, 8), scene_size)


@pytest.mark.parametrize('order', permutations((0, 1, 2, 3)))
def test_3d_slice_of_4d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small cube when displayed in 3D.
    """
    image = Image(np.zeros((16, 8, 4, 2)), scale=(1, 2, 4, 8))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(Dims(ndim=4, ndisplay=3, order=order))

    scene_size = vispy_image_scene_size(vispy_image)
    np.testing.assert_array_equal((16, 16, 16), scene_size)


def test_no_float32_texture_support(monkeypatch):
    """Ensure Image node can be created if OpenGL driver lacks float textures.

    See #3988, #3990, #6652.
    """
    monkeypatch.setattr(
        'napari._vispy.layers.image.get_gl_extensions', lambda: ''
    )
    image = Image(np.zeros((16, 8, 4, 2), dtype='uint8'), scale=(1, 2, 4, 8))
    VispyImageLayer(image)


@pytest.mark.parametrize('scale', ((1, 1, 1), (2, 2, 2)))
@pytest.mark.parametrize('ndisplay', (2, 3))
def test_node_transform_with_multiscale_then_consistent(scale, ndisplay):
    """See https://github.com/napari/napari/issues/6320"""
    # Define multi-scale image data with two levels where the
    # higher resolution is twice as high as the lower resolution.
    image = Image(data=[np.zeros((8, 8, 8)), np.zeros((4, 4, 4))], scale=scale)
    vispy_image = VispyImageLayer(image)

    # Take a full slice at the highest resolution.
    image.corner_pixels = np.array([[0, 0, 0], [8, 8, 8]])
    image._data_level = 0
    image._slice_dims(Dims(ndim=3, ndisplay=ndisplay, point=(1, 0, 0)))
    # Map the node's data origin to a vispy scene coordinate.
    high_res_origin = vispy_image.node.transform.map((0, 0))

    # Take a full slice at the lowest resolution and map the origin again.
    image.corner_pixels = np.array([[0, 0, 0], [4, 4, 4]])
    image._data_level = 1
    image._slice_dims(Dims(ndim=3, ndisplay=ndisplay, point=(1, 0, 0)))
    low_res_origin = vispy_image.node.transform.map((0, 0))

    # The exact origin may depend on certain parameter values, but the
    # full high and low resolution slices should always map to the same
    # scene origin, since this defines the start of the visible extent.
    np.testing.assert_array_equal(high_res_origin, low_res_origin)
