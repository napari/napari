import numpy as np
import pytest
from vispy.color import Colormap

from ..colormaps.colormaps import (
    AVAILABLE_COLORMAPS,
    increment_unnamed_colormap,
)


@pytest.mark.parametrize("name", list(AVAILABLE_COLORMAPS.keys()))
def test_colormap(name):
    np.random.seed(0)

    cmap = AVAILABLE_COLORMAPS[name]

    # Test can map random 0-1 values
    values = np.random.rand((50))
    colors = cmap.map(values)
    assert colors.shape == (len(values), 4)

    # Create vispy colormap and check current colormaps match vispy
    # colormap
    vispy_cmap = Colormap(*cmap)
    vispy_colors = vispy_cmap.map(values)
    np.testing.assert_almost_equal(colors, vispy_colors, decimal=6)


def test_increment_unnamed_colormap():
    # test that unnamed colormaps are incremented
    names = [
        '[unnamed colormap 0',
        'existing_colormap',
        'perceptually_uniform',
        '[unnamed colormap 1]',
    ]
    assert increment_unnamed_colormap(names) == '[unnamed colormap 2]'

    # test that named colormaps are not incremented
    named_colormap = 'perfect_colormap'
    assert increment_unnamed_colormap(names, named_colormap) == named_colormap
