import numpy as np

from napari.layers.labels.labels_utils import interpolate_coordinates


def test_interpolate_coordinates():
    # test when number of interpolated points > 1
    old_coord = np.array([0, 1])
    new_coord = np.array([0, 10])
    coords = interpolate_coordinates(old_coord, new_coord, brush_size=3)
    expected_coords = np.array(
        [
            [0, 1.75],
            [0, 2.5],
            [0, 3.25],
            [0, 4],
            [0, 4.75],
            [0, 5.5],
            [0, 6.25],
            [0, 7],
            [0, 7.75],
            [0, 8.5],
            [0, 9.25],
            [0, 10],
        ]
    )
    assert np.all(coords == expected_coords)
