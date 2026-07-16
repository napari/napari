import numpy as np
import pytest

from napari.utils.colormaps._tests.colors_data import (
    invalid_colors,
    single_color_options,
    single_colors_as_array,
    two_color_options,
    two_colors_as_array,
    warning_colors,
)
from napari.utils.colormaps.standardize_color import transform_color

COLORS = ['', (43, 3, 3, 3), np.array([[3, 3, 3, 3], [0, 0, 0, 1]])]

TRUE_COLORS = [
    np.zeros((1, 4), dtype=np.float32),
    np.array([[1, 3 / 43, 3 / 43, 3 / 43]], dtype=np.float32),
    np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
]
WARN_MESSAGE = (
    'Empty string detected',
    'Colors with values larger than one detected',
    'Colors with values larger than one detected',
)


@pytest.mark.parametrize(
    ('colors', 'true_colors'),
    zip(single_color_options, single_colors_as_array, strict=True),
)
def test_oned_points(colors, true_colors):
    np.testing.assert_array_equal(true_colors, transform_color(colors))


@pytest.mark.parametrize(
    ('color', 'true_color', 'warn_message'),
    zip(COLORS, TRUE_COLORS, WARN_MESSAGE, strict=True),
)
def test_warns_but_parses(color, true_color, warn_message):
    """Test collection of colors that raise a warning but do not return
    a default white color array.
    """

    with pytest.warns(UserWarning, match=warn_message):
        np.testing.assert_array_equal(true_color, transform_color(color))


@pytest.mark.parametrize(
    ('colors', 'true_colors'),
    zip(two_color_options, two_colors_as_array, strict=False),
)
def test_twod_points(colors, true_colors):
    np.testing.assert_array_equal(true_colors, transform_color(colors))


@pytest.mark.parametrize('color', invalid_colors)
def test_invalid_colors(color):
    with pytest.raises((ValueError, AttributeError, KeyError)):
        transform_color(color)


@pytest.mark.parametrize(('colors', 'warn_text'), warning_colors)
def test_warning_colors(colors, warn_text):
    with pytest.warns(UserWarning, match=warn_text):
        res = transform_color(colors)
    np.testing.assert_array_equal(
        np.ones((max(len(colors), 1), 4), dtype=np.float32), res
    )
