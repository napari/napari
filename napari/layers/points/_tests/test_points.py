from copy import copy
from xml.etree.ElementTree import Element

import numpy as np
import pandas as pd
import pytest
from vispy.color import get_colormap

from napari.layers import Points
from napari.utils.colormaps.standardize_color import transform_color


def test_empty_points():
    pts = Points()
    assert pts.data.shape == (0, 2)


def test_random_points():
    """Test instantiating Points layer with random 2D data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert np.all(layer.data == data)
    assert layer.ndim == shape[1]
    assert layer._view_data.ndim == 2
    assert len(layer.data) == 10
    assert len(layer.selected_data) == 0


def test_integer_points():
    """Test instantiating Points layer with integer data."""
    shape = (10, 2)
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 2))
    layer = Points(data)
    assert np.all(layer.data == data)
    assert layer.ndim == shape[1]
    assert layer._view_data.ndim == 2
    assert len(layer.data) == 10


def test_negative_points():
    """Test instantiating Points layer with negative data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape) - 10
    layer = Points(data)
    assert np.all(layer.data == data)
    assert layer.ndim == shape[1]
    assert layer._view_data.ndim == 2
    assert len(layer.data) == 10


def test_empty_points_array():
    """Test instantiating Points layer with empty array."""
    shape = (0, 2)
    data = np.empty(shape)
    layer = Points(data)
    assert np.all(layer.data == data)
    assert layer.ndim == shape[1]
    assert layer._view_data.ndim == 2
    assert len(layer.data) == 0


def test_3D_points():
    """Test instantiating Points layer with random 3D data."""
    shape = (10, 3)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert np.all(layer.data == data)
    assert layer.ndim == shape[1]
    assert layer._view_data.ndim == 2
    assert len(layer.data) == 10


def test_4D_points():
    """Test instantiating Points layer with random 4D data."""
    shape = (10, 4)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert np.all(layer.data == data)
    assert layer.ndim == shape[1]
    assert layer._view_data.ndim == 2
    assert len(layer.data) == 10


def test_changing_points():
    """Test changing Points data."""
    shape_a = (10, 2)
    shape_b = (20, 2)
    np.random.seed(0)
    data_a = 20 * np.random.random(shape_a)
    data_b = 20 * np.random.random(shape_b)
    layer = Points(data_a)
    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.ndim == shape_b[1]
    assert layer._view_data.ndim == 2
    assert len(layer.data) == 20


def test_selecting_points():
    """Test selecting points."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    layer.mode = 'select'
    data_to_select = [1, 2]
    layer.selected_data = data_to_select
    assert layer.selected_data == data_to_select

    # test switching to 3D
    layer.dims.ndisplay = 3
    assert layer.selected_data == data_to_select

    # select different points while in 3D mode
    other_data_to_select = [0]
    layer.selected_data = other_data_to_select
    assert layer.selected_data == other_data_to_select

    # selection should persist when going back to 2D mode
    layer.dims.ndisplay = 2
    assert layer.selected_data == other_data_to_select

    # selection should persist when switching between between select and pan_zoom
    layer.mode = 'pan_zoom'
    assert layer.selected_data == other_data_to_select
    layer.mode = 'select'
    assert layer.selected_data == other_data_to_select

    # add mode should clear the selection
    layer.mode = 'add'
    assert layer.selected_data == []


def test_adding_points():
    """Test adding Points data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert len(layer.data) == 10

    coord = [20, 20]
    layer.add(coord)
    assert len(layer.data) == 11
    assert np.all(layer.data[10] == coord)
    # the added point should be selected
    assert layer.selected_data == [10]

    # test adding multiple points
    coords = [[10, 10], [15, 15]]
    layer.add(coords)
    assert len(layer.data) == 13
    assert np.all(layer.data[11:, :] == coords)


def test_adding_points_to_empty():
    """Test adding Points data to empty."""
    shape = (0, 2)
    data = np.empty(shape)
    layer = Points(data)
    assert len(layer.data) == 0

    coord = [20, 20]
    layer.add(coord)
    assert len(layer.data) == 1
    assert np.all(layer.data[0] == coord)


def test_removing_selected_points():
    """Test selecting points."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)

    # With nothing selected no points should be removed
    layer.remove_selected()
    assert len(layer.data) == shape[0]

    # Select two points and remove them
    layer.selected_data = [0, 3]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 2
    assert len(layer.selected_data) == 0
    keep = [1, 2] + list(range(4, 10))
    assert np.all(layer.data == data[keep])

    # Select another point and remove it
    layer.selected_data = [4]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 3


def test_move():
    """Test moving points."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    unmoved = copy(data)
    layer = Points(data)

    # Move one point relative to an initial drag start location
    layer._move([0], [0, 0])
    layer._move([0], [10, 10])
    layer._drag_start = None
    assert np.all(layer.data[0] == unmoved[0] + [10, 10])
    assert np.all(layer.data[1:] == unmoved[1:])

    # Move two points relative to an initial drag start location
    layer._move([1, 2], [2, 2])
    layer._move([1, 2], np.add([2, 2], [-3, 4]))
    assert np.all(layer.data[1:2] == unmoved[1:2] + [-3, 4])


def test_changing_modes():
    """Test changing modes."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.mode == 'pan_zoom'
    assert layer.interactive is True

    layer.mode = 'add'
    assert layer.mode == 'add'
    assert layer.interactive is False

    layer.mode = 'select'
    assert layer.mode == 'select'
    assert layer.interactive is False

    layer.mode = 'pan_zoom'
    assert layer.mode == 'pan_zoom'
    assert layer.interactive is True


def test_name():
    """Test setting layer name."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    layer = Points(data)
    assert layer.name == 'Points'

    layer = Points(data, name='random')
    assert layer.name == 'random'

    layer.name = 'pts'
    assert layer.name == 'pts'


def test_visiblity():
    """Test setting layer visiblity."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    layer = Points(data)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Points(data, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_opacity():
    """Test setting layer opacity."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    layer = Points(data)
    assert layer.opacity == 1.0

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer = Points(data, opacity=0.6)
    assert layer.opacity == 0.6

    layer.opacity = 0.3
    assert layer.opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    layer = Points(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Points(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


def test_symbol():
    """Test setting symbol."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.symbol == 'disc'

    layer.symbol = 'cross'
    assert layer.symbol == 'cross'

    layer = Points(data, symbol='star')
    assert layer.symbol == 'star'


def test_properties():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'point_type': np.array(['A', 'B'] * int(shape[0] / 2))}
    layer = Points(data, properties=copy(properties))
    assert layer.properties == properties

    current_prop = {'point_type': np.array(['B'])}
    assert layer.current_properties == current_prop

    # test removing points
    layer.selected_data = [0, 1]
    layer.remove_selected()
    remove_properties = properties['point_type'][2::]
    assert len(layer.properties['point_type']) == (shape[0] - 2)
    assert np.all(layer.properties['point_type'] == remove_properties)

    # test selection of properties
    layer.selected_data = [0]
    selected_annotation = layer.current_properties['point_type']
    assert len(selected_annotation) == 1
    assert selected_annotation[0] == 'A'

    # test adding properties
    layer.add([10, 10])
    add_annotations = np.concatenate((remove_properties, ['A']), axis=0)
    assert np.all(layer.properties['point_type'] == add_annotations)

    # test copy/paste
    layer.selected_data = [0, 1]
    layer._copy_data()
    assert np.all(layer._clipboard['properties']['point_type'] == ['A', 'B'])

    layer._paste_data()
    paste_annotations = np.concatenate((add_annotations, ['A', 'B']), axis=0)
    assert np.all(layer.properties['point_type'] == paste_annotations)


def test_properties_dataframe():
    """test if properties can be provided as a DataFrame"""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'point_type': np.array(['A', 'B'] * int(shape[0] / 2))}
    properties_df = pd.DataFrame(properties)
    properties_df = properties_df.astype(properties['point_type'].dtype)
    layer = Points(data, properties=properties_df)
    np.testing.assert_equal(layer.properties, properties)


def test_adding_annotations():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'point_type': np.array(['A', 'B'] * int((shape[0] / 2)))}
    layer = Points(data)
    assert layer.properties == {}

    # add properties
    layer.properties = copy(properties)
    assert layer.properties == properties

    # change properties
    new_annotations = {
        'other_type': np.array(['C', 'D'] * int((shape[0] / 2)))
    }
    layer.properties = copy(new_annotations)
    assert layer.properties == new_annotations


def test_add_points_with_properties():
    # test adding points initialized with properties
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'point_type': np.array(['A', 'B'] * int((shape[0] / 2)))}
    layer = Points(data, properties=copy(properties))

    coord = [18, 18]
    layer.add(coord)
    new_prop = {'point_type': np.append(properties['point_type'], 'B')}
    np.testing.assert_equal(layer.properties, new_prop)


def test_annotations_errors():
    shape = (3, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)

    # try adding properties with the wrong number of properties
    with pytest.raises(ValueError):
        annotations = {'point_type': np.array(['A', 'B'])}
        Points(data, properties=copy(annotations))


def test_is_color_mapped():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    annotations = {'point_type': np.array(['A', 'B'] * int((shape[0] / 2)))}
    layer = Points(data, properties=annotations)

    # giving the name of an annotation should return True
    assert layer._is_color_mapped('point_type')

    # giving a list should return false (i.e., could be an RGBA color)
    assert not layer._is_color_mapped([1, 1, 1, 1])

    # giving an ndarray should return false (i.e., could be an RGBA color)
    assert not layer._is_color_mapped(np.array([1, 1, 1, 1]))

    # give an invalid color argument
    with pytest.raises(ValueError):
        layer._is_color_mapped((123, 323))


def test_edge_width():
    """Test setting edge width."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.edge_width == 1

    layer.edge_width = 2
    assert layer.edge_width == 2

    layer = Points(data, edge_width=3)
    assert layer.edge_width == 3


def test_n_dimensional():
    """Test setting n_dimensional flag for 2D and 4D data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.n_dimensional is False

    layer.n_dimensional = True
    assert layer.n_dimensional is True

    layer = Points(data, n_dimensional=True)
    assert layer.n_dimensional is True

    shape = (10, 4)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.n_dimensional is False

    layer.n_dimensional = True
    assert layer.n_dimensional is True

    layer = Points(data, n_dimensional=True)
    assert layer.n_dimensional is True


def test_edge_color_direct():
    """Test setting edge color."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    colorarray = transform_color(['black'] * shape[0])
    assert layer.current_edge_color == 'black'
    assert len(layer.edge_color) == shape[0]
    np.testing.assert_allclose(colorarray, layer.edge_color)

    # With no data selected chaning edge color has no effect
    layer.current_edge_color = 'blue'
    assert layer.current_edge_color == 'blue'
    np.testing.assert_allclose(colorarray, layer.edge_color)

    # Select data and change edge color of selection
    layer.selected_data = [0, 1]
    assert layer.current_edge_color == 'black'
    layer.current_edge_color = 'green'
    colorarray_green = transform_color(['green'] * len(layer.selected_data))
    np.testing.assert_allclose(colorarray_green, layer.edge_color[:2])
    np.testing.assert_allclose(colorarray[2:], layer.edge_color[2:])

    # Add new point and test its color
    coord = [18, 18]
    layer.selected_data = []
    layer.current_edge_color = 'blue'
    layer.add(coord)
    colorarray = np.vstack([colorarray, transform_color('blue')])
    assert len(layer.edge_color) == shape[0] + 1
    np.testing.assert_allclose(colorarray_green, layer.edge_color[:2])
    np.testing.assert_allclose(colorarray[2:], layer.edge_color[2:])
    np.testing.assert_allclose(
        transform_color("blue"), np.atleast_2d(layer.edge_color[10])
    )

    # Instantiate with custom edge color
    layer = Points(data, edge_color='red')
    assert layer.current_edge_color == 'red'

    # Instantiate with custom edge color list
    col_list = ['red', 'green'] * 5
    col_list_arr = transform_color(col_list)
    layer = Points(data, edge_color=col_list)
    assert layer.current_edge_color == 'green'
    np.testing.assert_allclose(layer.edge_color, col_list_arr)

    # Add new point and test its color
    coord = [18, 18]
    layer.current_edge_color = 'blue'
    layer.add(coord)
    assert len(layer.edge_color) == shape[0] + 1
    np.testing.assert_allclose(
        layer.edge_color, np.vstack((col_list_arr, transform_color('blue')))
    )

    # Check removing data adjusts colors correctly
    layer.selected_data = [0, 2]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.edge_color) == shape[0] - 1
    np.testing.assert_allclose(
        layer.edge_color,
        np.vstack(
            (col_list_arr[1], col_list_arr[3:], transform_color('blue'))
        ),
    )


def test_edge_color_cycle():
    # create Points using list color cycle
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    annotations = {'point_type': np.array(['A', 'B'] * int((shape[0] / 2)))}
    color_cycle = ['red', 'blue']
    layer = Points(
        data,
        properties=annotations,
        edge_color='point_type',
        edge_color_cycle=color_cycle,
    )
    assert layer.properties == annotations
    edge_color_array = transform_color(color_cycle * int((shape[0] / 2)))
    assert np.all(layer.edge_color == edge_color_array)

    # create Points using color array color cycle
    color_cycle_array = transform_color(color_cycle)
    layer2 = Points(
        data,
        properties=annotations,
        edge_color='point_type',
        edge_color_cycle=color_cycle_array,
    )
    assert np.all(layer2.edge_color == edge_color_array)

    # Add new point and test its color
    coord = [18, 18]
    layer2.selected_data = [0]
    layer2.add(coord)
    assert len(layer2.edge_color) == shape[0] + 1
    np.testing.assert_allclose(
        layer2.edge_color,
        np.vstack((edge_color_array, transform_color('red'))),
    )

    # Check removing data adjusts colors correctly
    layer2.selected_data = [0, 2]
    layer2.remove_selected()
    assert len(layer2.data) == shape[0] - 1
    assert len(layer2.edge_color) == shape[0] - 1
    np.testing.assert_allclose(
        layer2.edge_color,
        np.vstack(
            (edge_color_array[1], edge_color_array[3:], transform_color('red'))
        ),
    )


def test_edge_color_colormap():
    # create Points using with face_color colormap
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    annotations = {'point_type': np.array([0, 1.5] * int((shape[0] / 2)))}
    layer = Points(
        data,
        properties=annotations,
        edge_color='point_type',
        edge_colormap='gray',
    )
    assert layer.properties == annotations
    assert layer.edge_color_mode == 'colormap'
    edge_color_array = transform_color(
        ['black', 'white'] * int((shape[0] / 2))
    )
    assert np.all(layer.edge_color == edge_color_array)

    # change the color cycle - face_color should not change
    layer.edge_color_cycle = ['red', 'blue']
    assert np.all(layer.edge_color == edge_color_array)

    # Add new point and test its color
    coord = [18, 18]
    layer.selected_data = [0]
    layer.add(coord)
    assert len(layer.edge_color) == shape[0] + 1
    np.testing.assert_allclose(
        layer.edge_color,
        np.vstack((edge_color_array, transform_color('black'))),
    )

    # change the colormap
    new_colormap = 'viridis'
    layer.edge_colormap = new_colormap
    assert layer.edge_colormap[1] == get_colormap(new_colormap)

    # Check removing data adjusts colors correctly
    layer.selected_data = [0, 2]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.edge_color) == shape[0] - 1
    np.testing.assert_allclose(
        layer.edge_color,
        np.vstack(
            (
                edge_color_array[1],
                edge_color_array[3:],
                transform_color('black'),
            )
        ),
    )


def test_face_color_direct():
    """Test setting face color."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    colorarray = transform_color(['white'] * shape[0])
    assert layer.current_face_color == 'white'
    assert len(layer.face_color) == shape[0]
    np.testing.assert_allclose(colorarray, layer.face_color)

    # With no data selected chaning face color has no effect
    layer.current_face_color = 'blue'
    assert layer.current_face_color == 'blue'
    np.testing.assert_allclose(colorarray, layer.face_color)

    # Select data and change edge color of selection
    layer.selected_data = [0, 1]
    assert layer.current_face_color == 'white'
    layer.current_face_color = transform_color('green')
    colorarray_green = transform_color(['green'] * len(layer.selected_data))
    np.testing.assert_allclose(colorarray_green, layer.face_color[:2])
    np.testing.assert_allclose(colorarray[2:], layer.face_color[2:])

    # Add new point and test its color
    coord = [18, 18]
    layer.selected_data = []
    layer.current_face_color = 'blue'
    layer.add(coord)
    colorarray = np.vstack((colorarray, transform_color('blue')))
    assert len(layer.face_color) == shape[0] + 1
    np.testing.assert_allclose(colorarray_green, layer.face_color[:2])
    np.testing.assert_allclose(colorarray[2:], layer.face_color[2:])
    np.testing.assert_allclose(
        transform_color("blue"), np.atleast_2d(layer.face_color[10])
    )

    # Instantiate with custom face color
    layer = Points(data, face_color='red')
    assert layer.current_face_color == 'red'

    # Instantiate with custom face color list
    col_list = transform_color(['red', 'green'] * 5)
    layer = Points(data, face_color=col_list)
    assert layer.current_face_color == 'green'
    np.testing.assert_allclose(layer.face_color, col_list)

    # Add new point and test its color
    coord = [18, 18]
    layer.current_face_color = 'blue'
    layer.add(coord)
    assert len(layer.face_color) == shape[0] + 1
    np.testing.assert_allclose(
        layer.face_color, np.vstack((col_list, transform_color('blue')))
    )

    # Check removing data adjusts colors correctly
    layer.selected_data = [0, 2]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.face_color) == shape[0] - 1
    np.testing.assert_allclose(
        layer.face_color,
        np.vstack((col_list[1], col_list[3:], transform_color('blue'))),
    )


def test_face_color_cycle():
    # create Points using list color cycle
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    annotations = {'point_type': np.array(['A', 'B'] * int((shape[0] / 2)))}
    color_cycle = ['red', 'blue']
    layer = Points(
        data,
        properties=annotations,
        face_color='point_type',
        face_color_cycle=color_cycle,
    )
    assert layer.properties == annotations
    face_color_array = transform_color(color_cycle * int((shape[0] / 2)))
    assert np.all(layer.face_color == face_color_array)

    # create Points using color array color cycle
    color_cycle_array = transform_color(color_cycle)
    layer2 = Points(
        data,
        properties=annotations,
        face_color='point_type',
        face_color_cycle=color_cycle_array,
    )
    assert np.all(layer2.face_color == face_color_array)

    # Add new point and test its color
    coord = [18, 18]
    layer2.selected_data = [0]
    layer2.add(coord)
    assert len(layer2.face_color) == shape[0] + 1
    np.testing.assert_allclose(
        layer2.face_color,
        np.vstack((face_color_array, transform_color('red'))),
    )

    # Check removing data adjusts colors correctly
    layer2.selected_data = [0, 2]
    layer2.remove_selected()
    assert len(layer2.data) == shape[0] - 1
    assert len(layer2.face_color) == shape[0] - 1
    np.testing.assert_allclose(
        layer2.face_color,
        np.vstack(
            (face_color_array[1], face_color_array[3:], transform_color('red'))
        ),
    )


def test_face_color_colormap():
    # create Points using with face_color colormap
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    annotations = {'point_type': np.array([0, 1.5] * int((shape[0] / 2)))}
    layer = Points(
        data,
        properties=annotations,
        face_color='point_type',
        face_colormap='gray',
    )
    assert layer.properties == annotations
    assert layer.face_color_mode == 'colormap'
    face_color_array = transform_color(
        ['black', 'white'] * int((shape[0] / 2))
    )
    assert np.all(layer.face_color == face_color_array)

    # change the color cycle - face_color should not change
    layer.face_color_cycle = ['red', 'blue']
    assert np.all(layer.face_color == face_color_array)

    # Add new point and test its color
    coord = [18, 18]
    layer.selected_data = [0]
    layer.add(coord)
    assert len(layer.face_color) == shape[0] + 1
    np.testing.assert_allclose(
        layer.face_color,
        np.vstack((face_color_array, transform_color('black'))),
    )

    # change the colormap
    new_colormap = 'viridis'
    layer.face_colormap = new_colormap
    assert layer.face_colormap[1] == get_colormap(new_colormap)

    # Check removing data adjusts colors correctly
    layer.selected_data = [0, 2]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.face_color) == shape[0] - 1
    np.testing.assert_allclose(
        layer.face_color,
        np.vstack(
            (
                face_color_array[1],
                face_color_array[3:],
                transform_color('black'),
            )
        ),
    )


def test_size():
    """Test setting size with scalar."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.current_size == 10
    assert layer.size.shape == shape
    assert np.unique(layer.size)[0] == 10

    # Add a new point, it should get current size
    coord = [17, 17]
    layer.add(coord)
    assert layer.size.shape == (11, 2)
    assert np.unique(layer.size)[0] == 10

    # Setting size affects newly added points not current points
    layer.current_size = 20
    assert layer.current_size == 20
    assert layer.size.shape == (11, 2)
    assert np.unique(layer.size)[0] == 10

    # Add new point, should have new size
    coord = [18, 18]
    layer.add(coord)
    assert layer.size.shape == (12, 2)
    assert np.unique(layer.size[:11])[0] == 10
    assert np.all(layer.size[11] == [20, 20])

    # Select data and change size
    layer.selected_data = [0, 1]
    assert layer.current_size == 10
    layer.current_size = 16
    assert layer.size.shape == (12, 2)
    assert np.unique(layer.size[2:11])[0] == 10
    assert np.unique(layer.size[:2])[0] == 16

    # Select data and size changes
    layer.selected_data = [11]
    assert layer.current_size == 20

    # Create new layer with new size data
    layer = Points(data, size=15)
    assert layer.current_size == 15
    assert layer.size.shape == shape
    assert np.unique(layer.size)[0] == 15


def test_size_with_arrays():
    """Test setting size with arrays."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    sizes = 5 * np.random.random(shape)
    layer.size = sizes
    assert np.all(layer.size == sizes)

    # Test broadcasting of sizes
    sizes = [5, 5]
    layer.size = sizes
    assert np.all(layer.size[0] == sizes)

    # Create new layer with new size array data
    sizes = 5 * np.random.random(shape)
    layer = Points(data, size=sizes)
    assert layer.current_size == 10
    assert layer.size.shape == shape
    assert np.all(layer.size == sizes)

    # Create new layer with new size array data
    sizes = [5, 5]
    layer = Points(data, size=sizes)
    assert layer.current_size == 10
    assert layer.size.shape == shape
    assert np.all(layer.size[0] == sizes)

    # Add new point, should have new size
    coord = [18, 18]
    layer.current_size = 13
    layer.add(coord)
    assert layer.size.shape == (11, 2)
    assert np.unique(layer.size[:10])[0] == 5
    assert np.all(layer.size[10] == [13, 13])

    # Select data and change size
    layer.selected_data = [0, 1]
    assert layer.current_size == 5
    layer.current_size = 16
    assert layer.size.shape == (11, 2)
    assert np.unique(layer.size[2:10])[0] == 5
    assert np.unique(layer.size[:2])[0] == 16

    # Check removing data adjusts colors correctly
    layer.selected_data = [0, 2]
    layer.remove_selected()
    assert len(layer.data) == 9
    assert len(layer.size) == 9
    assert np.all(layer.size[0] == [16, 16])
    assert np.all(layer.size[1] == [5, 5])


def test_size_with_3D_arrays():
    """Test setting size with 3D arrays."""
    shape = (10, 3)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[:2, 0] = 0
    layer = Points(data)
    assert layer.current_size == 10
    assert layer.size.shape == shape
    assert np.unique(layer.size)[0] == 10

    sizes = 5 * np.random.random(shape)
    layer.size = sizes
    assert np.all(layer.size == sizes)

    # Test broadcasting of sizes
    sizes = [1, 5, 5]
    layer.size = sizes
    assert np.all(layer.size[0] == sizes)

    # Create new layer with new size array data
    sizes = 5 * np.random.random(shape)
    layer = Points(data, size=sizes)
    assert layer.current_size == 10
    assert layer.size.shape == shape
    assert np.all(layer.size == sizes)

    # Create new layer with new size array data
    sizes = [1, 5, 5]
    layer = Points(data, size=sizes)
    assert layer.current_size == 10
    assert layer.size.shape == shape
    assert np.all(layer.size[0] == sizes)

    # Add new point, should have new size in last dim only
    coord = [4, 18, 18]
    layer.current_size = 13
    layer.add(coord)
    assert layer.size.shape == (11, 3)
    assert np.unique(layer.size[:10, 1:])[0] == 5
    assert np.all(layer.size[10] == [1, 13, 13])

    # Select data and change size
    layer.selected_data = [0, 1]
    assert layer.current_size == 5
    layer.current_size = 16
    assert layer.size.shape == (11, 3)
    assert np.unique(layer.size[2:10, 1:])[0] == 5
    assert np.all(layer.size[0] == [16, 16, 16])

    # Create new 3D layer with new 2D points size data
    sizes = [0, 5, 5]
    layer = Points(data, size=sizes)
    assert layer.current_size == 10
    assert layer.size.shape == shape
    assert np.all(layer.size[0] == sizes)

    # Add new point, should have new size only in last 2 dimensions
    coord = [4, 18, 18]
    layer.current_size = 13
    layer.add(coord)
    assert layer.size.shape == (11, 3)
    assert np.all(layer.size[10] == [0, 13, 13])

    # Select data and change size
    layer.selected_data = [0, 1]
    assert layer.current_size == 5
    layer.current_size = 16
    assert layer.size.shape == (11, 3)
    assert np.unique(layer.size[2:10, 1:])[0] == 5
    assert np.all(layer.size[0] == [0, 16, 16])


def test_copy_and_paste():
    """Test copying and pasting selected points."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    # Clipboard starts empty
    assert layer._clipboard == {}

    # Pasting empty clipboard doesn't change data
    layer._paste_data()
    assert len(layer.data) == 10

    # Copying with nothing selected leave clipboard empty
    layer._copy_data()
    assert layer._clipboard == {}

    # Copying and pasting with two points selected adds to clipboard and data
    layer.selected_data = [0, 1]
    layer._copy_data()
    layer._paste_data()
    assert len(layer._clipboard.keys()) > 0
    assert len(layer.data) == shape[0] + 2
    assert np.all(layer.data[:2] == layer.data[-2:])

    # Pasting again adds two more points to data
    layer._paste_data()
    assert len(layer.data) == shape[0] + 4
    assert np.all(layer.data[:2] == layer.data[-2:])

    # Unselecting everything and copying and pasting will empty the clipboard
    # and add no new data
    layer.selected_data = []
    layer._copy_data()
    layer._paste_data()
    assert layer._clipboard == {}
    assert len(layer.data) == shape[0] + 4


def test_value():
    """Test getting the value of the data at the current coordinates."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[-1] = [0, 0]
    layer = Points(data)
    value = layer.get_value()
    assert layer.coordinates == (0, 0)
    assert value == 9

    layer.data = layer.data + 20
    value = layer.get_value()
    assert value is None


def test_message():
    """Test converting value and coords to message."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[-1] = [0, 0]
    layer = Points(data)
    msg = layer.get_message()
    assert type(msg) == str


def test_thumbnail():
    """Test the image thumbnail for square data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[0] = [0, 0]
    data[-1] = [20, 20]
    layer = Points(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_thumbnail_with_n_points_greater_than_max():
    """Test thumbnail generation with n_points > _max_points_thumbnail

    see: https://github.com/napari/napari/pull/934
    """
    # 2D
    max_points = Points._max_points_thumbnail * 2
    bigger_data = np.random.randint(10, 100, (max_points, 2))
    big_layer = Points(bigger_data)
    big_layer._update_thumbnail()
    assert big_layer.thumbnail.shape == big_layer._thumbnail_shape

    # #3D
    bigger_data_3d = np.random.randint(10, 100, (max_points, 3))
    bigger_layer_3d = Points(bigger_data_3d)
    bigger_layer_3d.dims.ndisplay = 3
    bigger_layer_3d._update_thumbnail()
    assert bigger_layer_3d.thumbnail.shape == bigger_layer_3d._thumbnail_shape


def test_xml_list():
    """Test the xml generation."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    xml = layer.to_xml_list()
    assert type(xml) == list
    assert len(xml) == shape[0]
    assert np.all([type(x) == Element for x in xml])


def test_view_data():
    coords = np.array([[0, 1, 1], [0, 2, 2], [1, 3, 3], [3, 3, 3]])
    layer = Points(coords)

    layer.dims.set_point(0, 0)
    assert np.all(
        layer._view_data == coords[np.ix_([0, 1], layer.dims.displayed)]
    )

    layer.dims.set_point(0, 1)
    assert np.all(
        layer._view_data == coords[np.ix_([2], layer.dims.displayed)]
    )

    layer.dims.ndisplay = 3
    assert np.all(layer._view_data == coords)


def test_view_size():
    coords = np.array([[0, 1, 1], [0, 2, 2], [1, 3, 3], [3, 3, 3]])
    sizes = np.array([[3, 5, 5], [3, 5, 5], [3, 3, 3], [2, 2, 3]])
    layer = Points(coords, size=sizes, n_dimensional=False)

    layer.dims.set_point(0, 0)
    assert np.all(
        layer._view_size == sizes[np.ix_([0, 1], layer.dims.displayed)]
    )

    layer.dims.set_point(0, 1)
    assert np.all(layer._view_size == sizes[np.ix_([2], layer.dims.displayed)])

    layer.n_dimensional = True
    assert len(layer._view_size) == 3


def test_view_colors():
    coords = [[0, 1, 1], [0, 2, 2], [1, 3, 3], [3, 3, 3]]
    face_color = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
    )
    edge_color = np.array(
        [[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    )

    layer = Points(coords, face_color=face_color, edge_color=edge_color)
    layer.dims.set_point(0, 0)
    print(layer.face_color)
    print(layer._view_face_color)
    assert np.all(layer._view_face_color == face_color[[0, 1]])
    assert np.all(layer._view_edge_color == edge_color[[0, 1]])

    layer.dims.set_point(0, 1)
    assert np.all(layer._view_face_color == face_color[[2]])
    assert np.all(layer._view_edge_color == edge_color[[2]])

    # view colors should return empty array if there are no points
    layer.dims.set_point(0, 2)
    assert len(layer._view_face_color) == 0
    assert len(layer._view_edge_color) == 0
