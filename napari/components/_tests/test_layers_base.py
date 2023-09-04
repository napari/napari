from napari.layers.base import Layer
import pytest
import numpy as np
from numpy import array


@pytest.mark.parametrize(
    'dims,nworld,nshape,expected',
    [
        ([2, 1, 0, 3], 4, 2, [0, 1]),
        ([2, 1, 0, 3], 4, 3, [1, 0, 2]),
        ([2, 1, 0, 3], 4, 4, [2, 1, 0, 3]),
        ([0, 1, 2, 3, 4, 5, 6, 7], 4, 4, [0, 1, 2, 3, 4, 5, 6, 7]),
    ],
)
def test_world_to_layer(dims, nworld, nshape, expected):
    assert np.array_equal(
        Layer._world_to_layer_dims_impl(array(dims), nworld, nshape), expected
    )
