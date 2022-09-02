import numpy as np
import pytest

from napari._vispy.layers.image import VispyImageLayer
from napari.layers import Image


@pytest.mark.parametrize('order', ((0, 1, 2), (2, 1, 0), (0, 2, 1)))
def test_3d_slice_of_3d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926"""
    image = Image(np.zeros((4, 4, 2)), scale=(1, 1, 2))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(ndisplay=3, order=order)

    node = vispy_image.node
    world_size = node.transform.map(node._last_data.shape)
    np.testing.assert_array_equal((4, 4, 4, 1), world_size)
