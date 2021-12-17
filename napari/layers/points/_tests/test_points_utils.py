import numpy as np

from napari.layers.points._points_utils import (
    _create_box_from_corners_3d,
    _points_in_box_3d,
)


def test_create_box_from_corners_3d():
    corners = np.array([[5, 0, 0], [5, 10, 10]])
    normal = np.array([1, 0, 0])
    up_dir = np.array([0, 1, 0])

    box = _create_box_from_corners_3d(
        box_corners=corners, box_normal=normal, up_vector=up_dir
    )

    expected_box = np.array([[5, 0, 0], [5, 0, 10], [5, 10, 10], [5, 10, 0]])
    np.testing.assert_allclose(box, expected_box)


def test_points_in_box_3d():
    normal = np.array([1, 0, 0])
    up_dir = np.array([0, 1, 0])
    corners = np.array([[10, 10, 10], [10, 20, 20]])
    points = np.array([[0, 15, 15], [10, 30, 25], [10, 12, 18], [20, 15, 30]])
    sizes = np.ones((points.shape[0],))

    inside = _points_in_box_3d(
        box_corners=corners,
        box_normal=normal,
        up_direction=up_dir,
        points=points,
        sizes=sizes,
    )

    assert set(inside) == {0, 2}
