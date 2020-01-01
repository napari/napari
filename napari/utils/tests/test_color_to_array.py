import pytest
import numpy as np

from .colors_data import (
    single_color_options,
    single_colors_as_array,
    two_color_options,
    two_colors_as_array,
    invalid_colors,
    warning_colors,
)
from napari.utils.colormaps.standardize_color import transform_color


@pytest.mark.parametrize(
    "colors, true_colors", zip(single_color_options, single_colors_as_array),
)
def test_oned_points(colors, true_colors):
    np.testing.assert_array_equal(true_colors, transform_color(colors))


@pytest.mark.parametrize(
    "colors, true_colors", zip(two_color_options, two_colors_as_array)
)
def test_twod_points(colors, true_colors):
    np.testing.assert_array_equal(true_colors, transform_color(colors))


@pytest.mark.parametrize("color", invalid_colors)
def test_invalid_colors(color):
    with pytest.raises((ValueError, AttributeError, KeyError)):
        transform_color(color)


@pytest.mark.parametrize("colors", warning_colors)
def test_warning_colors(colors):
    with pytest.warns(UserWarning):
        np.testing.assert_array_equal(
            np.ones((len(colors), 4), dtype=np.float32),
            transform_color(colors),
        )
