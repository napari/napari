import numpy as np

from napari.layers.points._points_utils import create_box_from_corners_3d


def test_create_box_from_corners_3d():
    corners = np.array([[5, 0, 0], [5, 10, 10]])
    normal = np.array([1, 0, 0])

    box = create_box_from_corners_3d(corners, normal)

    expected_box = np.array([[5, 0, 0], [5, 0, 10], [5, 10, 10], [5, 10, 0]])
    np.testing.assert_allclose(box, expected_box)
