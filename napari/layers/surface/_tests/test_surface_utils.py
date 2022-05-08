import numpy as np
import pytest

from napari.layers.surface._surface_utils import (
    calculate_barycentric_coordinates,
)


@pytest.mark.parametrize(
    "point,expected_barycentric_coordinates",
    [
        ([5, 1, 1], [1 / 3, 1 / 3, 1 / 3]),
        ([5, 0, 0], [1, 0, 0]),
        ([5, 0, 3], [0, 1, 0]),
        ([5, 3, 0], [0, 0, 1]),
    ],
)
def test_calculate_barycentric_coordinates(
    point, expected_barycentric_coordinates
):
    triangle_vertices = np.array(
        [
            [5, 0, 0],
            [5, 0, 3],
            [5, 3, 0],
        ]
    )
    u, v, w = calculate_barycentric_coordinates(point, triangle_vertices)
    np.testing.assert_allclose([u, v, w], expected_barycentric_coordinates)
    np.testing.assert_allclose(u + v + w, 1)
