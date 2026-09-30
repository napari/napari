import numpy as np
import pytest

from napari.utils.color import (
    _contrast_ratio,
    _readable_color,
    rgb_to_luminance,
)


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


def test_contrast_ratio():
    assert _contrast_ratio([1, 1, 1], [0, 0, 0]) == pytest.approx(21)
    assert _contrast_ratio([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) == 1


def test_readable_color_keeps_hue():
    white = np.array([1.0, 1.0, 1.0, 1.0])
    readable = _readable_color([0, 1, 0, 1], white)
    assert _contrast_ratio(readable, white) >= 4.5
    assert readable[1] == readable[:3].max()

    black = np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        _readable_color([0, 1, 0, 1], black), [0, 1, 0, 1]
    )
