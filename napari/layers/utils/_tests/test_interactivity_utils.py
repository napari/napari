import numpy as np
import pytest

from napari.layers import Image
from napari.layers.utils.interactivity_utils import (
    drag_data_to_projected_distance,
    orient_plane_normal_around_cursor,
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


@pytest.mark.parametrize(
    'layer',
    [
        Image(np.zeros(shape=(28, 28, 28))),
        Image(np.zeros(shape=(2, 28, 28, 28))),
    ],
)
def test_orient_plane_normal_around_cursor(make_napari_viewer, layer):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (0, 0, 90)
    viewer.cursor.position = [14] * layer._ndim

    viewer.add_layer(layer)
    layer.depiction = 'plane'
    layer.plane.normal = (1, 0, 0)
    layer.plane.position = (14, 14, 14)

    # apply simple transformation on the volume
    layer.translate = [1] * layer._ndim

    # orient plane normal
    orient_plane_normal_around_cursor(layer=layer, plane_normal=(1, 0, 1))

    # check that plane normal has been updated
    assert np.allclose(
        layer.plane.normal, [1, 0, 1] / np.linalg.norm([1, 0, 1])
    )
    assert np.allclose(layer.plane.position, (14, 13, 13))
