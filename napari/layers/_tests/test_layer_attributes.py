import numpy as np
import pytest

from napari._tests.utils import layer_test_data
from napari.layers import Image


@pytest.mark.parametrize(
    'image_shape, dims_displayed, expected',
    [
        ((10, 20, 30), (0, 1, 2), [[0, 10], [0, 20], [0, 30]]),
        ((10, 20, 30), (0, 2, 1), [[0, 10], [0, 30], [0, 20]]),
        ((10, 20, 30), (2, 1, 0), [[0, 30], [0, 20], [0, 10]]),
    ],
)
def test_layer_bounding_box_order(image_shape, dims_displayed, expected):
    layer = Image(data=np.random.random(image_shape))
    #
    assert np.allclose(
        layer._display_bounding_box(dims_displayed=dims_displayed), expected
    )


@pytest.mark.parametrize('Layer, data, ndim', layer_test_data)
def test_layer_extent_cache(Layer, data, ndim):
    np.random.seed(0)
    layer = Layer(data)
    # Check layer has been correctly created
    assert layer.ndim == ndim

    # Check layer extent doesn't change when unrelated attribute changes
    layer.opacity = 1
    old_extent = layer.extent
    assert old_extent is layer.extent
    layer.opacity = 0.5
    assert old_extent is layer.extent

    # Check layer extent change when scale changes
    layer.scale = (2,) + layer.scale[1:]
    old_extent2 = layer.extent
    assert old_extent is not layer.extent
    assert old_extent2 is layer.extent
