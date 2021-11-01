import numpy as np
import pytest

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
