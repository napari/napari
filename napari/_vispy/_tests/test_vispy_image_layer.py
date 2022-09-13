from itertools import permutations
from typing import Tuple, Union

import numpy as np
import pytest
from vispy.visuals import ImageVisual, VolumeVisual
from vispy.visuals.transforms.linear import STTransform

from napari._vispy.layers.image import VispyImageLayer
from napari.layers import Image


def _node_scene_size(
    node: Union[ImageVisual, VolumeVisual]
) -> Tuple[float, float, float, float]:
    """Calculates the size of a vispy node in 3D scene homogeneous coordinates."""
    data = node._last_data if isinstance(node, VolumeVisual) else node._data
    # Only use scale to ignore translate offset used to center top-left pixel.
    transform = STTransform(scale=np.diag(node.transform.matrix))
    # Vispy uses an xy-style ordering, whereas numpy uses a rc-style
    # ordering, so reverse the shape before applying the transform.
    return transform.map(data.shape[::-1])


@pytest.mark.parametrize('order', permutations((0, 1, 2)))
def test_3d_slice_of_2d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small square when displayed in 3D.
    """
    image = Image(np.zeros((4, 2)), scale=(1, 2))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(point=(0, 0, 0), ndisplay=3, order=order)

    scene_size = _node_scene_size(vispy_image.node)
    np.testing.assert_array_equal((4, 4, 1, 1), scene_size)


@pytest.mark.parametrize('order', permutations((0, 1, 2)))
def test_2d_slice_of_3d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small square when displayed in 2D.
    """
    image = Image(np.zeros((8, 4, 2)), scale=(1, 2, 4))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(point=(0, 0, 0), ndisplay=2, order=order)

    scene_size = _node_scene_size(vispy_image.node)
    np.testing.assert_array_equal((8, 8, 0, 1), scene_size)


@pytest.mark.parametrize('order', permutations((0, 1, 2)))
def test_3d_slice_of_3d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small cube when displayed in 3D.
    """
    image = Image(np.zeros((8, 4, 2)), scale=(1, 2, 4))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(point=(0, 0, 0), ndisplay=3, order=order)

    scene_size = _node_scene_size(vispy_image.node)
    np.testing.assert_array_equal((8, 8, 8, 1), scene_size)


@pytest.mark.parametrize('order', permutations((0, 1, 2, 3)))
def test_3d_slice_of_4d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small cube when displayed in 3D.
    """
    image = Image(np.zeros((16, 8, 4, 2)), scale=(1, 2, 4, 8))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(point=(0, 0, 0, 0), ndisplay=3, order=order)

    scene_size = _node_scene_size(vispy_image.node)
    np.testing.assert_array_equal((16, 16, 16, 1), scene_size)
