import pytest

from .colors_data import (
    single_color_options,
    single_colors_as_colorarray,
    two_color_options,
    two_colors_as_colorarray,
    invalid_colors,
)
from napari.util.color.standardize_color import transform_color


@pytest.mark.parametrize(
    "colors, true_colors",
    zip(single_color_options, single_colors_as_colorarray),
)
def test_oned_points(colors, true_colors):
    assert true_colors == transform_color(colors)


@pytest.mark.parametrize(
    "colors, true_colors", zip(two_color_options, two_colors_as_colorarray)
)
def test_twod_points(colors, true_colors):
    assert true_colors == transform_color(colors)


@pytest.mark.parametrize("color", invalid_colors)
def test_invalid_colors(color):
    with pytest.raises((ValueError, AttributeError, KeyError)):
        transform_color(color)
