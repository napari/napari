import pytest
import numpy as np

from ..colormaps.colormaps import AVAILABLE_COLORMAPS
from vispy.color.color_array import ColorArray


@pytest.mark.parametrize("name", list(AVAILABLE_COLORMAPS.keys()))
def test_colormap(name):
    cmap = AVAILABLE_COLORMAPS[name]

    # colormaps should accept a scalar with the __getitem__ method
    # and return a ColorArray
    assert isinstance(cmap[0.5], ColorArray)

    # colormaps should accept a 1D array with the __getitem__ method
    # and return a ColorArray
    assert isinstance(cmap[np.linspace(0, 1, 256) ** 0.5], ColorArray)

    # colormap.map() is a lower level API
    # it takes a (N, 1) vector of values in [0, 1], and returns a rgba array of
    # size (N, 4). as per the vispy documentation: This function doesn't need
    # to implement argument checking on `item`. It can always assume that
    # `item` is a (N, 1) array of values between 0 and 1.
    # http://vispy.org/color.html
    q = np.random.rand(10, 10)
    assert cmap.map(q.reshape(-1, 1)).shape == (q.size, 4)
