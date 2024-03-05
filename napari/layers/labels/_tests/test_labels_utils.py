import numpy as np

from napari.components.dims import Dims
from napari.layers.labels import Labels
from napari.layers.labels._labels_utils import (
    first_nonzero_coordinate,
    get_dtype,
    interpolate_coordinates,
    mouse_event_to_labels_coordinate,
)
from napari.utils._proxies import ReadOnlyWrapper


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
    np.testing.assert_array_equal(coords, expected_coords)


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


def test_mouse_event_to_labels_coordinate_2d(MouseEvent):
    data = np.zeros((11, 11), dtype=int)
    data[4:7, 4:7] = 1
    layer = Labels(data, scale=(2, 2))

    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(10, 10),
            view_direction=None,
            dims_displayed=(1, 2),
            dims_point=(0, 0),
        )
    )
    coord = mouse_event_to_labels_coordinate(layer, event)
    np.testing.assert_array_equal(coord, [5, 5])


def test_mouse_event_to_labels_coordinate_3d(MouseEvent):
    data = np.zeros((11, 11, 11), dtype=int)
    data[4:7, 4:7, 4:7] = 1
    layer = Labels(data, scale=(2, 2, 2))
    layer._slice_dims(Dims(ndim=3, ndisplay=3))

    # click straight down from the top
    # (note the scale on the layer!)
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(0, 10, 10),
            view_direction=(1, 0, 0),
            dims_displayed=(0, 1, 2),
            dims_point=(10, 10, 10),
        )
    )
    coord = mouse_event_to_labels_coordinate(layer, event)
    np.testing.assert_array_equal(coord, [4, 5, 5])

    # click diagonally from the top left corner
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(0.1, 0, 0),
            view_direction=np.full(3, 1 / np.sqrt(3)),
            dims_displayed=(0, 1, 2),
            dims_point=(10, 10, 10),
        )
    )
    coord = mouse_event_to_labels_coordinate(layer, event)
    np.testing.assert_array_equal(coord, [4, 4, 4])

    # drag starts inside volume but ends up outside volume
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_press',
            is_dragging=True,
            position=(-100, -100, -100),
            view_direction=(1, 0, 0),
            dims_displayed=(0, 1, 2),
            dims_point=(10, 10, 10),
        )
    )

    coord = mouse_event_to_labels_coordinate(layer, event)
    assert coord is None
