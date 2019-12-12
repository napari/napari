import pytest
import numpy as np

from vispy.color import ColorArray
from colors_data import *  # noqa: F403
from napari.layers.util.standardize_color import transform_color


@pytest.mark.parametrize("colors, true_colors", zip(single_color_options, single_colors_as_colorarray))
def test_oned_points(colors, true_colors):
    assert true_colors == transform_color(colors)


@pytest.mark.parametrize("colors, true_colors", zip(two_color_options, two_colors_as_colorarray))
def test_twod_points(colors, true_colors):
    assert true_colors == transform_color(colors)
