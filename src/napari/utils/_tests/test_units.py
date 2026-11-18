import numpy as np
import pytest

from napari.utils._units import compute_nice_ticks


@pytest.mark.parametrize(
    ('start', 'stop', 'n_ticks', 'expected_ticks', 'expected_decimals'),
    [
        (0, 3, 5, [0, 0.75, 1.5, 2.25, 3], 2),
        (0, 5, 5, [0, 1.5, 3, 4.5], 1),
        (0, 3, 10, [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3], 1),
        (-1, 1, 5, [-1, -0.5, 0, 0.5, 1], 1),
        (0, 0.03, 5, [0, 0.0075, 0.015, 0.0225, 0.03], 4),
        (101985, 339015, 5, [150000, 200000, 250000, 300000], 0),
        (1, 1, 5, [], 0),
    ],
)
def test_compute_nice_ticks(
    start, stop, n_ticks, expected_ticks, expected_decimals
):
    ticks, decimals = compute_nice_ticks(start, stop, n_ticks)
    np.testing.assert_allclose(ticks, expected_ticks)
    assert decimals == expected_decimals
