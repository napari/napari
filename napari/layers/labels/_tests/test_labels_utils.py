import numpy as np

from napari.layers.labels import Labels
from napari.layers.labels._labels_utils import (
    first_nonzero_coordinate,
    get_dtype,
    interpolate_coordinates,
)


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


def test_interpolate_with_none():
    """Test that interpolating with one None coordinate returns original."""
    coord = np.array([5, 5])
    expected = coord[np.newaxis, :]
    actual = interpolate_coordinates(coord, None, brush_size=1)
    np.testing.assert_array_equal(actual, expected)
    actual2 = interpolate_coordinates(None, coord, brush_size=5)
    np.testing.assert_array_equal(actual2, expected)


def test_get_dtype():
    np.random.seed(0)
    data = np.random.randint(20, size=(50, 50))
    layer = Labels(data)

    assert get_dtype(layer) == data.dtype

    data2 = data[::2, ::2]
    layer_data = [data, data2]
    multiscale_layer = Labels(layer_data)
    assert get_dtype(multiscale_layer) == layer_data[0].dtype

    data = data.astype(int)
    int_layer = Labels(data)
    assert get_dtype(int_layer) == int


def test_first_nonzero_coordinate():
    data = np.zeros((11, 11, 11))
    data[4:7, 4:7, 4:7] = 1
    np.testing.assert_array_equal(
        first_nonzero_coordinate(data, np.zeros(3), np.full(3, 10)),
        [4, 4, 4],
    )
    np.testing.assert_array_equal(
        first_nonzero_coordinate(data, np.full(3, 10), np.zeros(3)),
        [6, 6, 6],
    )
    assert (
        first_nonzero_coordinate(data, np.zeros(3), np.array([0, 1, 1]))
        is None
    )
    np.testing.assert_array_equal(
        first_nonzero_coordinate(
            data, np.array([0, 6, 6]), np.array([10, 5, 5])
        ),
        [4, 6, 6],
    )
