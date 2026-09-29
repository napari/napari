import numpy as np
import pytest

from napari.utils.color import rgb_to_luminance


@pytest.mark.parametrize(
    ('rgb', 'luminance'),
    [
        ([1, 1, 1], 1),
        ([0, 0, 0], 0),
        ([1, 1, 0], 0.9278),
        ([[1, 1, 0], [0, 0, 1]], [0.9278, 0.0722]),
        # works with alpha
        ([[1, 1, 0, 1], [0, 0, 1, 0.5]], [0.9278, 0.0361]),
        # works with arbitrarily high values
        ([[2, 2, 0, 2], [0, 0, 1, 0.5]], [0.9278 * 4, 0.0361]),
        # works with arbitrary shapes
        (np.ones((10, 10, 10, 3)), np.ones((10, 10, 10))),
        (np.ones((10, 10, 10, 4)), np.ones((10, 10, 10))),
    ],
)
def test_rgb_to_luminance(rgb, luminance):
    np.testing.assert_array_almost_equal(
        rgb_to_luminance(np.asarray(rgb)), luminance
    )
