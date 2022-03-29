import numpy as np
import pytest

from napari.layers.utils.interactivity_utils import (
    drag_data_to_projected_distance,
)


@pytest.mark.parametrize(
    "start_position, end_position, view_direction, vector, expected_value",
    [
        # drag vector parallel to view direction
        # projected onto perpendicular vector
        ([0, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], 0),
        # same as above, projection onto multiple perpendicular vectors
        # should produce multiple results
        ([0, 0, 0], [0, 0, 1], [0, 0, 1], [[1, 0, 0], [0, 1, 0]], [0, 0]),
        # drag vector perpendicular to view direction
        # projected onto itself
        ([0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], 1),
        # drag vector perpendicular to view direction
        # projected onto itself
        ([0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], 1),
    ],
)
def test_drag_data_to_projected_distance(
    start_position, end_position, view_direction, vector, expected_value
):
    result = drag_data_to_projected_distance(
        start_position, end_position, view_direction, vector
    )
    assert np.allclose(result, expected_value)
