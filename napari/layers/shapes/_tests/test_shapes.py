from copy import copy
from itertools import cycle, islice

import numpy as np
import pandas as pd
import pytest

from napari._tests.utils import check_layer_world_data_extent
from napari.layers import Shapes
from napari.utils.colormaps.standardize_color import transform_color


def _make_cycled_properties(values, length):
    """Helper function to make property values

    Parameters
    ----------
    values
        The values to be cycled.
    length : int
        The length of the resulting property array

    Returns
    -------
    cycled_properties : np.ndarray
        The property array comprising the cycled values.
    """
    cycled_properties = np.array(list(islice(cycle(values), 0, length)))
    return cycled_properties


def test_empty_shapes():
    shp = Shapes()
    assert shp.ndim == 2


properties_array = {'shape_type': _make_cycled_properties(['A', 'B'], 10)}
properties_list = {'shape_type': list(_make_cycled_properties(['A', 'B'], 10))}


@pytest.mark.parametrize("properties", [properties_array, properties_list])
def test_properties(properties):
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, properties=copy(properties))
    np.testing.assert_equal(layer.properties, properties)

    current_prop = {'shape_type': np.array(['B'])}
    assert layer.current_properties == current_prop

    # test removing shapes
    layer.selected_data = {0, 1}
    layer.remove_selected()
    remove_properties = properties['shape_type'][2::]
    assert len(layer.properties['shape_type']) == (shape[0] - 2)
    assert np.all(layer.properties['shape_type'] == remove_properties)

    # test selection of properties
    layer.selected_data = {0}
    selected_annotation = layer.current_properties['shape_type']
    assert len(selected_annotation) == 1
    assert selected_annotation[0] == 'A'

    # test adding shapes with properties
    new_data = np.random.random((1, 4, 2))
    new_shape_type = ['rectangle']
    layer.add(new_data, shape_type=new_shape_type)
    add_properties = np.concatenate((remove_properties, ['A']), axis=0)
    assert np.all(layer.properties['shape_type'] == add_properties)

    # test copy/paste
    layer.selected_data = {0, 1}
    layer._copy_data()
    assert np.all(layer._clipboard['properties']['shape_type'] == ['A', 'B'])

    layer._paste_data()
    paste_properties = np.concatenate((add_properties, ['A', 'B']), axis=0)
    assert np.all(layer.properties['shape_type'] == paste_properties)

    # test updating a property
    layer.mode = 'select'
    layer.selected_data = {0}
    new_property = {'shape_type': np.array(['B'])}
    layer.current_properties = new_property
    updated_properties = layer.properties
    assert updated_properties['shape_type'][0] == 'B'


@pytest.mark.parametrize("attribute", ['edge', 'face'])
def test_adding_properties(attribute):
    """Test adding properties to an existing layer"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)

    # add properties
    properties = {'shape_type': _make_cycled_properties(['A', 'B'], shape[0])}
    layer.properties = properties
    np.testing.assert_equal(layer.properties, properties)

    # add properties as a dataframe
    properties_df = pd.DataFrame(properties)
    layer.properties = properties_df
    np.testing.assert_equal(layer.properties, properties)

    # add properties as a dictionary with list values
    properties_list = {
        'shape_type': list(_make_cycled_properties(['A', 'B'], shape[0]))
    }
    layer.properties = properties_list
    assert isinstance(layer.properties['shape_type'], np.ndarray)

    # removing a property that was the _*_color_property should give a warning
    setattr(layer, f'_{attribute}_color_property', 'shape_type')
    properties_2 = {
        'not_shape_type': _make_cycled_properties(['A', 'B'], shape[0])
    }
    with pytest.warns(RuntimeWarning):
        layer.properties = properties_2


def test_data_setter_with_properties():
    """Test layer data on a layer with properties via the data setter"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'shape_type': _make_cycled_properties(['A', 'B'], shape[0])}
    layer = Shapes(data, properties=properties)

    # test setting to data with fewer shapes
    n_new_shapes = 4
    new_data = 20 * np.random.random((n_new_shapes, 4, 2))
    layer.data = new_data
    assert len(layer.properties['shape_type']) == n_new_shapes

    # test setting to data with more shapes
    n_new_shapes_2 = 6
    new_data_2 = 20 * np.random.random((n_new_shapes_2, 4, 2))
    layer.data = new_data_2
    assert len(layer.properties['shape_type']) == n_new_shapes_2

    # test setting to data with same shapes
    new_data_3 = 20 * np.random.random((n_new_shapes_2, 4, 2))
    layer.data = new_data_3
    assert len(layer.properties['shape_type']) == n_new_shapes_2


def test_properties_dataframe():
    """Test if properties can be provided as a DataFrame"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'shape_type': _make_cycled_properties(['A', 'B'], shape[0])}
    properties_df = pd.DataFrame(properties)
    properties_df = properties_df.astype(properties['shape_type'].dtype)
    layer = Shapes(data, properties=properties_df)
    np.testing.assert_equal(layer.properties, properties)


def test_empty_layer_with_text_properties():
    """Test initializing an empty layer with text defined"""
    default_properties = {'shape_type': np.array([1.5], dtype=float)}
    text_kwargs = {'text': 'shape_type', 'color': 'red'}
    layer = Shapes(
        properties=default_properties,
        text=text_kwargs,
    )
    assert layer.text.mode == 'property'
    np.testing.assert_equal(layer.text.values, np.empty(0))
    np.testing.assert_allclose(layer.text.color, [1, 0, 0, 1])

    # add a shape and check that the appropriate text value was added
    layer.add(np.random.random((1, 4, 2)))
    np.testing.assert_equal(layer.text.values, ['1.5'])
    np.testing.assert_allclose(layer.text.color, [1, 0, 0, 1])


def test_empty_layer_with_text_formatted():
    """Test initializing an empty layer with text defined"""
    default_properties = {'shape_type': np.array([1.5], dtype=float)}
    layer = Shapes(
        properties=default_properties,
        text='shape_type: {shape_type:.2f}',
    )
    assert layer.text.mode == 'formatted'
    np.testing.assert_equal(layer.text.values, np.empty(0))

    # add a shape and check that the appropriate text value was added
    layer.add(np.random.random((1, 4, 2)))
    np.testing.assert_equal(layer.text.values, ['shape_type: 1.50'])


@pytest.mark.parametrize("properties", [properties_array, properties_list])
def test_text_from_property_value(properties):
    """Test setting text from a property value"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, properties=copy(properties), text='shape_type')

    np.testing.assert_equal(layer.text.values, properties['shape_type'])


@pytest.mark.parametrize("properties", [properties_array, properties_list])
def test_text_from_property_fstring(properties):
    """Test setting text with an f-string from the property value"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(
        data, properties=copy(properties), text='type: {shape_type}'
    )

    expected_text = ['type: ' + v for v in properties['shape_type']]
    np.testing.assert_equal(layer.text.values, expected_text)

    # test updating the text
    layer.text = 'type-ish: {shape_type}'
    expected_text_2 = ['type-ish: ' + v for v in properties['shape_type']]
    np.testing.assert_equal(layer.text.values, expected_text_2)

    # copy/paste
    layer.selected_data = {0}
    layer._copy_data()
    layer._paste_data()
    expected_text_3 = expected_text_2 + ['type-ish: A']
    np.testing.assert_equal(layer.text.values, expected_text_3)

    # add shape
    layer.selected_data = {0}
    new_shape = np.random.random((1, 4, 2))
    layer.add(new_shape)
    expected_text_4 = expected_text_3 + ['type-ish: A']
    np.testing.assert_equal(layer.text.values, expected_text_4)


@pytest.mark.parametrize("properties", [properties_array, properties_list])
def test_set_text_with_kwarg_dict(properties):
    text_kwargs = {
        'text': 'type: {shape_type}',
        'color': [0, 0, 0, 1],
        'rotation': 10,
        'translation': [5, 5],
        'anchor': 'upper_left',
        'size': 10,
        'visible': True,
    }
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, properties=copy(properties), text=text_kwargs)

    expected_text = ['type: ' + v for v in properties['shape_type']]
    np.testing.assert_equal(layer.text.values, expected_text)

    for property, value in text_kwargs.items():
        if property == 'text':
            continue
        layer_value = getattr(layer._text, property)
        np.testing.assert_equal(layer_value, value)


@pytest.mark.parametrize("properties", [properties_array, properties_list])
def test_text_error(properties):
    """creating a layer with text as the wrong type should raise an error"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    # try adding text as the wrong type
    with pytest.raises(TypeError):
        Shapes(data, properties=copy(properties), text=123)


def test_refresh_text():
    """Test refreshing the text after setting new properties"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'shape_type': ['A'] * shape[0]}
    layer = Shapes(data, properties=copy(properties), text='shape_type')

    new_properties = {'shape_type': ['B'] * shape[0]}
    layer.properties = new_properties
    np.testing.assert_equal(layer.text.values, new_properties['shape_type'])


def test_nd_text():
    """Test slicing of text coords with nD shapes"""
    shapes_data = [
        [[0, 10, 10, 10], [0, 10, 20, 20], [0, 10, 10, 20], [0, 10, 20, 10]],
        [[1, 20, 30, 30], [1, 20, 50, 50], [1, 20, 50, 30], [1, 20, 30, 50]],
    ]
    properties = {'shape_type': ['A', 'B']}
    text_kwargs = {'text': 'shape_type', 'anchor': 'center'}
    layer = Shapes(shapes_data, properties=properties, text=text_kwargs)
    assert layer.ndim == 4

    layer._slice_dims(point=[0, 10, 0, 0], ndisplay=2)
    np.testing.assert_equal(layer._indices_view, [0])
    np.testing.assert_equal(layer._view_text_coords[0], [[15, 15]])

    layer._slice_dims(point=[1, 0, 0, 0], ndisplay=3)
    np.testing.assert_equal(layer._indices_view, [1])
    np.testing.assert_equal(layer._view_text_coords[0], [[20, 40, 40]])


@pytest.mark.parametrize("properties", [properties_array, properties_list])
def test_data_setter_with_text(properties):
    """Test layer data on a layer with text via the data setter"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, properties=copy(properties), text='shape_type')

    # test setting to data with fewer shapes
    n_new_shapes = 4
    new_data = 20 * np.random.random((n_new_shapes, 4, 2))
    layer.data = new_data
    assert len(layer.text.values) == n_new_shapes

    # test setting to data with more shapes
    n_new_shapes_2 = 6
    new_data_2 = 20 * np.random.random((n_new_shapes_2, 4, 2))
    layer.data = new_data_2
    assert len(layer.text.values) == n_new_shapes_2

    # test setting to data with same shapes
    new_data_3 = 20 * np.random.random((n_new_shapes_2, 4, 2))
    layer.data = new_data_3
    assert len(layer.text.values) == n_new_shapes_2


def test_rectangles():
    """Test instantiating Shapes layer with a random 2D rectangles."""
    # Test a single four corner rectangle
    shape = (1, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == data[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])

    # Test multiple four corner rectangles
    shape = (10, 4, 2)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])

    # Test a single two corner rectangle, which gets converted into four
    # corner rectangle
    shape = (1, 2, 2)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.nshapes == 1
    assert len(layer.data[0]) == 4
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])

    # Test multiple two corner rectangles
    shape = (10, 2, 2)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([len(ld) == 4 for ld in layer.data])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_rectangles_with_shape_type():
    """Test instantiating rectangles with shape_type in data"""
    # Test (rectangle, shape_type) tuple
    shape = (1, 4, 2)
    np.random.seed(0)
    vertices = 20 * np.random.random(shape)
    data = (vertices, "rectangle")
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == data[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])

    # Test (list of rectangles, shape_type) tuple
    shape = (10, 4, 2)
    vertices = 20 * np.random.random(shape)
    data = (vertices, "rectangle")
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, vertices)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])

    # Test list of (rectangle, shape_type) tuples
    data = [(vertices[i], "rectangle") for i in range(shape[0])]
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, vertices)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_rectangles_roundtrip():
    """Test a full roundtrip with rectangles data."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    new_layer = Shapes(layer.data)
    assert np.all([nd == d for nd, d in zip(new_layer.data, layer.data)])


def test_integer_rectangle():
    """Test instantiating rectangles with integer data."""
    shape = (10, 2, 2)
    np.random.seed(1)
    data = np.random.randint(20, size=shape)
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([len(ld) == 4 for ld in layer.data])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_negative_rectangle():
    """Test instantiating rectangles with negative data."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape) - 10
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_empty_rectangle():
    """Test instantiating rectangles with empty data."""
    shape = (0, 0, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_3D_rectangles():
    """Test instantiating Shapes layer with 3D planar rectangles."""
    # Test a single four corner rectangle
    np.random.seed(0)
    planes = np.tile(np.arange(10).reshape((10, 1, 1)), (1, 4, 1))
    corners = np.random.uniform(0, 10, size=(10, 4, 2))
    data = np.concatenate((planes, corners), axis=2)
    layer = Shapes(data)
    assert layer.nshapes == len(data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == 3
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_ellipses():
    """Test instantiating Shapes layer with a random 2D ellipses."""
    # Test a single four corner ellipses
    shape = (1, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='ellipse')
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == data[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    # Test multiple four corner ellipses
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='ellipse')
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    # Test a single ellipse center radii, which gets converted into four
    # corner ellipse
    shape = (1, 2, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='ellipse')
    assert layer.nshapes == 1
    assert len(layer.data[0]) == 4
    assert layer.ndim == shape[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    # Test multiple center radii ellipses
    shape = (10, 2, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='ellipse')
    assert layer.nshapes == shape[0]
    assert np.all([len(ld) == 4 for ld in layer.data])
    assert layer.ndim == shape[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_ellipses_with_shape_type():
    """Test instantiating ellipses with shape_type in data"""
    # Test single four corner (vertices, shape_type) tuple
    shape = (1, 4, 2)
    np.random.seed(0)
    vertices = 20 * np.random.random(shape)
    data = (vertices, "ellipse")
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == data[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    # Test multiple four corner (list of vertices, shape_type) tuple
    shape = (10, 4, 2)
    np.random.seed(0)
    vertices = 20 * np.random.random(shape)
    data = (vertices, "ellipse")
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, vertices)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    # Test list of four corner (vertices, shape_type) tuples
    shape = (10, 4, 2)
    np.random.seed(0)
    vertices = 20 * np.random.random(shape)
    data = [(vertices[i], "ellipse") for i in range(shape[0])]
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, vertices)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    # Test single (center-radii, shape_type) ellipse
    shape = (1, 2, 2)
    np.random.seed(0)
    data = (20 * np.random.random(shape), "ellipse")
    layer = Shapes(data)
    assert layer.nshapes == 1
    assert len(layer.data[0]) == 4
    assert layer.ndim == shape[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    # Test (list of center-radii, shape_type) tuple
    shape = (10, 2, 2)
    np.random.seed(0)
    center_radii = 20 * np.random.random(shape)
    data = (center_radii, "ellipse")
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([len(ld) == 4 for ld in layer.data])
    assert layer.ndim == shape[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    # Test list of (center-radii, shape_type) tuples
    shape = (10, 2, 2)
    np.random.seed(0)
    center_radii = 20 * np.random.random(shape)
    data = [(center_radii[i], "ellipse") for i in range(shape[0])]
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([len(ld) == 4 for ld in layer.data])
    assert layer.ndim == shape[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_4D_ellispse():
    """Test instantiating Shapes layer with 4D planar ellipse."""
    # Test a single 4D ellipse
    np.random.seed(0)
    data = [
        [
            [3, 5, 108, 108],
            [3, 5, 108, 148],
            [3, 5, 148, 148],
            [3, 5, 148, 108],
        ]
    ]
    layer = Shapes(data, shape_type='ellipse')
    assert layer.nshapes == len(data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == 4
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_ellipses_roundtrip():
    """Test a full roundtrip with ellipss data."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='ellipse')
    new_layer = Shapes(layer.data, shape_type='ellipse')
    assert np.all([nd == d for nd, d in zip(new_layer.data, layer.data)])


def test_lines():
    """Test instantiating Shapes layer with a random 2D lines."""
    # Test a single two end point line
    shape = (1, 2, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='line')
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == data[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'line' for s in layer.shape_type])

    # Test multiple lines
    shape = (10, 2, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='line')
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'line' for s in layer.shape_type])


def test_lines_with_shape_type():
    """Test instantiating lines with shape_type"""
    # Test (single line, shape_type) tuple
    shape = (1, 2, 2)
    np.random.seed(0)
    end_points = 20 * np.random.random(shape)
    data = (end_points, 'line')
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == end_points[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'line' for s in layer.shape_type])

    # Test (multiple lines, shape_type) tuple
    shape = (10, 2, 2)
    np.random.seed(0)
    end_points = 20 * np.random.random(shape)
    data = (end_points, "line")
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, end_points)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'line' for s in layer.shape_type])

    # Test list of (line, shape_type) tuples
    shape = (10, 2, 2)
    np.random.seed(0)
    end_points = 20 * np.random.random(shape)
    data = [(end_points[i], "line") for i in range(shape[0])]
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, end_points)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'line' for s in layer.shape_type])


def test_lines_roundtrip():
    """Test a full roundtrip with line data."""
    shape = (10, 2, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='line')
    new_layer = Shapes(layer.data, shape_type='line')
    assert np.all([nd == d for nd, d in zip(new_layer.data, layer.data)])


def test_paths():
    """Test instantiating Shapes layer with a random 2D paths."""
    # Test a single path with 6 points
    shape = (1, 6, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='path')
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == data[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'path' for s in layer.shape_type])

    # Test multiple paths with different numbers of points
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(10)
    ]
    layer = Shapes(data, shape_type='path')
    assert layer.nshapes == len(data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == 2
    assert np.all([s == 'path' for s in layer.shape_type])


def test_paths_with_shape_type():
    """Test instantiating paths with shape_type in data"""
    # Test (single path, shape_type) tuple
    shape = (1, 6, 2)
    np.random.seed(0)
    path_points = 20 * np.random.random(shape)
    data = (path_points, "path")
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == path_points[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'path' for s in layer.shape_type])

    # Test (list of paths, shape_type) tuple
    path_points = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(10)
    ]
    data = (path_points, "path")
    layer = Shapes(data)
    assert layer.nshapes == len(path_points)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, path_points)])
    assert layer.ndim == 2
    assert np.all([s == 'path' for s in layer.shape_type])

    # Test list of  (path, shape_type) tuples
    data = [(path_points[i], "path") for i in range(len(path_points))]
    layer = Shapes(data)
    assert layer.nshapes == len(data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, path_points)])
    assert layer.ndim == 2
    assert np.all([s == 'path' for s in layer.shape_type])


def test_paths_roundtrip():
    """Test a full roundtrip with path data."""
    np.random.seed(0)
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(10)
    ]
    layer = Shapes(data, shape_type='path')
    new_layer = Shapes(layer.data, shape_type='path')
    assert np.all(
        [np.all(nd == d) for nd, d in zip(new_layer.data, layer.data)]
    )


def test_polygons():
    """Test instantiating Shapes layer with a random 2D polygons."""
    # Test a single polygon with 6 points
    shape = (1, 6, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='polygon')
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == data[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'polygon' for s in layer.shape_type])

    # Test multiple polygons with different numbers of points
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(10)
    ]
    layer = Shapes(data, shape_type='polygon')
    assert layer.nshapes == len(data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == 2
    assert np.all([s == 'polygon' for s in layer.shape_type])


def test_polygons_with_shape_type():
    """Test 2D polygons with shape_type in data"""

    # Test single (polygon, shape_type) tuple
    shape = (1, 6, 2)
    np.random.seed(0)
    vertices = 20 * np.random.random(shape)
    data = (vertices, 'polygon')
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == vertices[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'polygon' for s in layer.shape_type])

    # Test (list of polygons, shape_type) tuple
    polygons = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(10)
    ]
    data = (polygons, 'polygon')
    layer = Shapes(data)
    assert layer.nshapes == len(polygons)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, polygons)])
    assert layer.ndim == 2
    assert np.all([s == 'polygon' for s in layer.shape_type])

    # Test list of (polygon, shape_type) tuples
    data = [(polygons[i], 'polygon') for i in range(len(polygons))]
    layer = Shapes(data)
    assert layer.nshapes == len(polygons)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, polygons)])
    assert layer.ndim == 2
    assert np.all([s == 'polygon' for s in layer.shape_type])


def test_polygon_roundtrip():
    """Test a full roundtrip with polygon data."""
    np.random.seed(0)
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(10)
    ]
    layer = Shapes(data, shape_type='polygon')
    new_layer = Shapes(layer.data, shape_type='polygon')
    assert np.all(
        [np.all(nd == d) for nd, d in zip(new_layer.data, layer.data)]
    )


def test_mixed_shapes():
    """Test instantiating Shapes layer with a mix of random 2D shapes."""
    # Test multiple polygons with different numbers of points
    np.random.seed(0)
    shape_vertices = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(5)
    ] + list(np.random.random((5, 4, 2)))
    shape_type = ['polygon'] * 5 + ['rectangle'] * 3 + ['ellipse'] * 2
    layer = Shapes(shape_vertices, shape_type=shape_type)
    assert layer.nshapes == len(shape_vertices)
    assert np.all(
        [np.all(ld == d) for ld, d in zip(layer.data, shape_vertices)]
    )
    assert layer.ndim == 2
    assert np.all([s == so for s, so in zip(layer.shape_type, shape_type)])

    # Test roundtrip with mixed data
    new_layer = Shapes(layer.data, shape_type=layer.shape_type)
    assert np.all(
        [np.all(nd == d) for nd, d in zip(new_layer.data, layer.data)]
    )
    assert np.all(
        [ns == s for ns, s in zip(new_layer.shape_type, layer.shape_type)]
    )


def test_mixed_shapes_with_shape_type():
    """Test adding mixed shapes with shape_type in data"""
    np.random.seed(0)
    shape_vertices = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(5)
    ] + list(np.random.random((5, 4, 2)))
    shape_type = ['polygon'] * 5 + ['rectangle'] * 3 + ['ellipse'] * 2

    # Test multiple (shape, shape_type) tuples
    data = list(zip(shape_vertices, shape_type))
    layer = Shapes(data)
    assert layer.nshapes == len(shape_vertices)
    assert np.all(
        [np.all(ld == d) for ld, d in zip(layer.data, shape_vertices)]
    )
    assert layer.ndim == 2
    assert np.all([s == so for s, so in zip(layer.shape_type, shape_type)])


def test_data_shape_type_overwrites_meta():
    """Test shape type passed through data property overwrites metadata shape type"""
    shape = (10, 4, 2)
    np.random.seed(0)
    vertices = 20 * np.random.random(shape)
    data = (vertices, "ellipse")
    layer = Shapes(data, shape_type='rectangle')
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    data = [(vertices[i], "ellipse") for i in range(shape[0])]
    layer = Shapes(data, shape_type='rectangle')
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_changing_shapes():
    """Test changing Shapes data."""
    shape_a = (10, 4, 2)
    shape_b = (20, 4, 2)
    np.random.seed(0)
    vertices_a = 20 * np.random.random(shape_a)
    vertices_b = 20 * np.random.random(shape_b)
    layer = Shapes(vertices_a)
    assert layer.nshapes == shape_a[0]
    layer.data = vertices_b
    assert layer.nshapes == shape_b[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, vertices_b)])
    assert layer.ndim == shape_b[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])

    # setting data with shape type
    data_a = (vertices_a, "ellipse")
    layer.data = data_a
    assert layer.nshapes == shape_a[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, vertices_a)])
    assert layer.ndim == shape_a[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    # setting data with fewer shapes
    smaller_data = vertices_a[:5]
    current_edge_color = layer._data_view.edge_color
    current_edge_width = layer._data_view.edge_widths
    current_face_color = layer._data_view.face_color
    current_z = layer._data_view.z_indices

    layer.data = smaller_data
    assert layer.nshapes == smaller_data.shape[0]
    assert np.allclose(layer._data_view.edge_color, current_edge_color[:5])
    assert np.allclose(layer._data_view.face_color, current_face_color[:5])
    assert np.allclose(layer._data_view.edge_widths, current_edge_width[:5])
    assert np.allclose(layer._data_view.z_indices, current_z[:5])

    # setting data with added shapes
    current_edge_color = layer._data_view.edge_color
    current_edge_width = layer._data_view.edge_widths
    current_face_color = layer._data_view.face_color
    current_z = layer._data_view.z_indices

    bigger_data = vertices_b
    layer.data = bigger_data
    assert layer.nshapes == bigger_data.shape[0]
    assert np.allclose(layer._data_view.edge_color[:5], current_edge_color)
    assert np.allclose(layer._data_view.face_color[:5], current_face_color)
    assert np.allclose(layer._data_view.edge_widths[:5], current_edge_width)
    assert np.allclose(layer._data_view.z_indices[:5], current_z)


def test_changing_shape_type():
    """Test changing shape type"""
    np.random.seed(0)
    rectangles = 20 * np.random.random((10, 4, 2))
    layer = Shapes(rectangles, shape_type='rectangle')
    layer.shape_type = "ellipse"
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_adding_shapes():
    """Test adding shapes."""
    # Start with polygons with different numbers of points
    np.random.seed(0)
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(5)
    ]
    # shape_type = ['polygon'] * 5 + ['rectangle'] * 3 + ['ellipse'] * 2
    layer = Shapes(data, shape_type='polygon')
    new_data = np.random.random((5, 4, 2))
    new_shape_type = ['rectangle'] * 3 + ['ellipse'] * 2
    layer.add(new_data, shape_type=new_shape_type)
    all_data = data + list(new_data)
    all_shape_type = ['polygon'] * 5 + new_shape_type
    assert layer.nshapes == len(all_data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, all_data)])
    assert layer.ndim == 2
    assert np.all([s == so for s, so in zip(layer.shape_type, all_shape_type)])

    # test adding data with shape_type
    new_vertices = np.random.random((5, 4, 2))
    new_shape_type2 = ['ellipse'] * 3 + ['rectangle'] * 2
    new_data2 = list(zip(new_vertices, new_shape_type2))
    layer.add(new_data2)
    all_vertices = all_data + list(new_vertices)
    all_shape_type = all_shape_type + new_shape_type2
    assert layer.nshapes == len(all_vertices)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, all_vertices)])
    assert layer.ndim == 2
    assert np.all([s == so for s, so in zip(layer.shape_type, all_shape_type)])


def test_adding_shapes_to_empty():
    """Test adding shapes to empty."""
    data = np.empty((0, 0, 2))
    np.random.seed(0)
    layer = Shapes(np.empty((0, 0, 2)))
    assert len(layer.data) == 0

    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(5)
    ] + list(np.random.random((5, 4, 2)))
    shape_type = ['path'] * 5 + ['rectangle'] * 3 + ['ellipse'] * 2

    layer.add(data, shape_type=shape_type)
    assert layer.nshapes == len(data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == 2
    assert np.all([s == so for s, so in zip(layer.shape_type, shape_type)])


def test_selecting_shapes():
    """Test selecting shapes."""
    data = 20 * np.random.random((10, 4, 2))
    np.random.seed(0)
    layer = Shapes(data)
    layer.selected_data = {0, 1}
    assert layer.selected_data == {0, 1}

    layer.selected_data = {9}
    assert layer.selected_data == {9}

    layer.selected_data = set()
    assert layer.selected_data == set()


def test_removing_all_shapes_empty_list():
    """Test removing all shapes with an empty list."""
    data = 20 * np.random.random((10, 4, 2))
    np.random.seed(0)
    layer = Shapes(data)
    assert layer.nshapes == 10

    layer.data = []
    assert layer.nshapes == 0


def test_removing_all_shapes_empty_array():
    """Test removing all shapes with an empty list."""
    data = 20 * np.random.random((10, 4, 2))
    np.random.seed(0)
    layer = Shapes(data)
    assert layer.nshapes == 10

    layer.data = np.empty((0, 2))
    assert layer.nshapes == 0


def test_removing_selected_shapes():
    """Test removing selected shapes."""
    np.random.seed(0)
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(5)
    ] + list(np.random.random((5, 4, 2)))
    shape_type = ['polygon'] * 5 + ['rectangle'] * 3 + ['ellipse'] * 2
    layer = Shapes(data, shape_type=shape_type)

    # With nothing selected no points should be removed
    layer.remove_selected()
    assert len(layer.data) == len(data)

    # Select three shapes and remove them
    layer.selected_data = {1, 7, 8}
    layer.remove_selected()
    keep = [0] + list(range(2, 7)) + [9]
    data_keep = [data[i] for i in keep]
    shape_type_keep = [shape_type[i] for i in keep]
    assert len(layer.data) == len(data_keep)
    assert len(layer.selected_data) == 0
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data_keep)])
    assert layer.ndim == 2
    assert np.all(
        [s == so for s, so in zip(layer.shape_type, shape_type_keep)]
    )


def test_changing_modes():
    """Test changing modes."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    layer = Shapes(data)
    assert layer.mode == 'pan_zoom'
    assert layer.interactive is True

    layer.mode = 'select'
    assert layer.mode == 'select'
    assert layer.interactive is False

    layer.mode = 'direct'
    assert layer.mode == 'direct'
    assert layer.interactive is False

    layer.mode = 'vertex_insert'
    assert layer.mode == 'vertex_insert'
    assert layer.interactive is False

    layer.mode = 'vertex_remove'
    assert layer.mode == 'vertex_remove'
    assert layer.interactive is False

    layer.mode = 'add_rectangle'
    assert layer.mode == 'add_rectangle'
    assert layer.interactive is False

    layer.mode = 'add_ellipse'
    assert layer.mode == 'add_ellipse'
    assert layer.interactive is False

    layer.mode = 'add_line'
    assert layer.mode == 'add_line'
    assert layer.interactive is False

    layer.mode = 'add_path'
    assert layer.mode == 'add_path'
    assert layer.interactive is False

    layer.mode = 'add_polygon'
    assert layer.mode == 'add_polygon'
    assert layer.interactive is False

    layer.mode = 'pan_zoom'
    assert layer.mode == 'pan_zoom'
    assert layer.interactive is True


def test_name():
    """Test setting layer name."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    layer = Shapes(data)
    assert layer.name == 'Shapes'

    layer = Shapes(data, name='random')
    assert layer.name == 'random'

    layer.name = 'shps'
    assert layer.name == 'shps'


def test_visiblity():
    """Test setting layer visibility."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    layer = Shapes(data)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Shapes(data, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_opacity():
    """Test setting opacity."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    # Check default opacity value of 0.7
    assert layer.opacity == 0.7

    # Select data and change opacity of selection
    layer.selected_data = {0, 1}
    assert layer.opacity == 0.7
    layer.opacity = 0.5
    assert layer.opacity == 0.5

    # Add new shape and test its width
    new_shape = np.random.random((1, 4, 2))
    layer.selected_data = set()
    layer.add(new_shape)
    assert layer.opacity == 0.5

    # Instantiate with custom opacity
    layer2 = Shapes(data, opacity=0.2)
    assert layer2.opacity == 0.2

    # Check removing data shouldn't change opacity
    layer2.selected_data = {0, 2}
    layer2.remove_selected()
    assert len(layer2.data) == shape[0] - 2
    assert layer2.opacity == 0.2


def test_blending():
    """Test setting layer blending."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    layer = Shapes(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Shapes(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


@pytest.mark.filterwarnings("ignore:elementwise comparison fail:FutureWarning")
@pytest.mark.parametrize("attribute", ['edge', 'face'])
def test_switch_color_mode(attribute):
    """Test switching between color modes"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    # create a continuous property with a known value in the last element
    continuous_prop = np.random.random((shape[0],))
    continuous_prop[-1] = 1
    properties = {
        'shape_truthiness': continuous_prop,
        'shape_type': _make_cycled_properties(['A', 'B'], shape[0]),
    }
    initial_color = [1, 0, 0, 1]
    color_cycle = ['red', 'blue']
    color_kwarg = f'{attribute}_color'
    colormap_kwarg = f'{attribute}_colormap'
    color_cycle_kwarg = f'{attribute}_color_cycle'
    args = {
        color_kwarg: initial_color,
        colormap_kwarg: 'gray',
        color_cycle_kwarg: color_cycle,
    }
    layer = Shapes(data, properties=properties, **args)

    layer_color_mode = getattr(layer, f'{attribute}_color_mode')
    layer_color = getattr(layer, f'{attribute}_color')
    assert layer_color_mode == 'direct'
    np.testing.assert_allclose(
        layer_color, np.repeat([initial_color], shape[0], axis=0)
    )

    # there should not be an edge_color_property
    color_property = getattr(layer, f'_{attribute}_color_property')
    assert color_property == ''

    # transitioning to colormap should raise a warning
    # because there isn't an edge color property yet and
    # the first property in shapes.properties is being automatically selected
    with pytest.warns(UserWarning):
        setattr(layer, f'{attribute}_color_mode', 'colormap')
    color_property = getattr(layer, f'_{attribute}_color_property')
    assert color_property == next(iter(properties))
    layer_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(layer_color[-1], [1, 1, 1, 1])

    # switch to color cycle
    setattr(layer, f'{attribute}_color_mode', 'cycle')
    setattr(layer, f'{attribute}_color', 'shape_type')
    color = getattr(layer, f'{attribute}_color')
    layer_color = transform_color(color_cycle * int(shape[0] / 2))
    np.testing.assert_allclose(color, layer_color)

    # switch back to direct, edge_colors shouldn't change
    setattr(layer, f'{attribute}_color_mode', 'direct')
    new_edge_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(new_edge_color, color)


@pytest.mark.parametrize("attribute", ['edge', 'face'])
def test_color_direct(attribute: str):
    """Test setting face/edge color directly."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer_kwargs = {f'{attribute}_color': 'black'}
    layer = Shapes(data, **layer_kwargs)
    color_array = transform_color(['black'] * shape[0])

    current_color = getattr(layer, f'current_{attribute}_color')
    layer_color = getattr(layer, f'{attribute}_color')
    assert current_color == 'black'
    assert len(layer.edge_color) == shape[0]
    np.testing.assert_allclose(color_array, layer_color)

    # With no data selected changing color has no effect
    setattr(layer, f'current_{attribute}_color', 'blue')
    current_color = getattr(layer, f'current_{attribute}_color')
    assert current_color == 'blue'
    np.testing.assert_allclose(color_array, layer_color)

    # Select data and change edge color of selection
    selected_data = {0, 1}
    layer.selected_data = {0, 1}
    current_color = getattr(layer, f'current_{attribute}_color')
    assert current_color == 'black'
    setattr(layer, f'current_{attribute}_color', 'green')
    colorarray_green = transform_color(['green'] * len(layer.selected_data))
    color_array[list(selected_data)] = colorarray_green
    layer_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(color_array, layer_color)
    # Add new shape and test its color
    new_shape = np.random.random((1, 4, 2))
    layer.selected_data = set()
    setattr(layer, f'current_{attribute}_color', 'blue')
    layer.add(new_shape)
    color_array = np.vstack([color_array, transform_color('blue')])
    layer_color = getattr(layer, f'{attribute}_color')
    assert len(layer_color) == shape[0] + 1
    np.testing.assert_allclose(color_array, layer_color)

    # Check removing data adjusts colors correctly
    layer.selected_data = {0, 2}
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1

    layer_color = getattr(layer, f'{attribute}_color')
    assert len(layer_color) == shape[0] - 1
    np.testing.assert_allclose(
        layer_color,
        np.vstack((color_array[1], color_array[3:])),
    )

    # set the color directly
    setattr(layer, f'{attribute}_color', 'black')
    color_array = np.tile([[0, 0, 0, 1]], (len(layer.data), 1))
    layer_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(color_array, layer_color)


@pytest.mark.parametrize("attribute", ['edge', 'face'])
def test_single_shape_properties(attribute):
    """Test creating single shape with properties"""
    shape = (4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer_kwargs = {f'{attribute}_color': 'red'}
    layer = Shapes(data, **layer_kwargs)
    layer_color = getattr(layer, f'{attribute}_color')
    assert len(layer_color) == 1
    np.testing.assert_allclose([1, 0, 0, 1], layer_color[0])


color_cycle_str = ['red', 'blue']
color_cycle_rgb = [[1, 0, 0], [0, 0, 1]]
color_cycle_rgba = [[1, 0, 0, 1], [0, 0, 1, 1]]


@pytest.mark.parametrize("attribute", ['edge', 'face'])
@pytest.mark.parametrize(
    "color_cycle",
    [color_cycle_str, color_cycle_rgb, color_cycle_rgba],
)
def test_color_cycle(attribute, color_cycle):
    """Test setting edge/face color with a color cycle list"""
    # create Shapes using list color cycle
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'shape_type': _make_cycled_properties(['A', 'B'], shape[0])}
    shapes_kwargs = {
        'properties': properties,
        f'{attribute}_color': 'shape_type',
        f'{attribute}_color_cycle': color_cycle,
    }
    layer = Shapes(data, **shapes_kwargs)

    assert layer.properties == properties
    color_array = transform_color(
        list(islice(cycle(color_cycle), 0, shape[0]))
    )
    layer_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(layer_color, color_array)

    # Add new shape and test its color
    new_shape = np.random.random((1, 4, 2))
    layer.selected_data = {0}
    layer.add(new_shape)
    layer_color = getattr(layer, f'{attribute}_color')
    assert len(layer_color) == shape[0] + 1
    np.testing.assert_allclose(
        layer_color,
        np.vstack((color_array, transform_color('red'))),
    )

    # Check removing data adjusts colors correctly
    layer.selected_data = {0, 2}
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1

    layer_color = getattr(layer, f'{attribute}_color')
    assert len(layer_color) == shape[0] - 1
    np.testing.assert_allclose(
        layer_color,
        np.vstack((color_array[1], color_array[3:], transform_color('red'))),
    )

    # refresh colors
    layer.refresh_colors(update_color_mapping=True)

    # test adding a shape with a new property value
    layer.selected_data = {}
    current_properties = layer.current_properties
    current_properties['shape_type'] = np.array(['new'])
    layer.current_properties = current_properties
    new_shape_2 = np.random.random((1, 4, 2))
    layer.add(new_shape_2)
    color_cycle_map = getattr(layer, f'{attribute}_color_cycle_map')

    assert 'new' in color_cycle_map
    np.testing.assert_allclose(
        color_cycle_map['new'], np.squeeze(transform_color(color_cycle[0]))
    )


@pytest.mark.parametrize("attribute", ['edge', 'face'])
def test_add_color_cycle_to_empty_layer(attribute):
    """Test adding a shape to an empty layer when edge/face color is a color cycle

    See: https://github.com/napari/napari/pull/1069
    """
    default_properties = {'shape_type': np.array(['A'])}
    color_cycle = ['red', 'blue']
    shapes_kwargs = {
        'properties': default_properties,
        f'{attribute}_color': 'shape_type',
        f'{attribute}_color_cycle': color_cycle,
    }
    layer = Shapes(**shapes_kwargs)

    # verify the current_edge_color is correct
    expected_color = transform_color(color_cycle[0])
    current_color = getattr(layer, f'_current_{attribute}_color')
    np.testing.assert_allclose(current_color, expected_color)

    # add a shape
    np.random.seed(0)
    new_shape = 20 * np.random.random((1, 4, 2))
    layer.add(new_shape)
    props = {'shape_type': np.array(['A'])}
    expected_color = np.array([[1, 0, 0, 1]])
    np.testing.assert_equal(layer.properties, props)
    attribute_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(attribute_color, expected_color)

    # add a shape with a new property
    layer.selected_data = []
    layer.current_properties = {'shape_type': np.array(['B'])}
    new_shape_2 = 20 * np.random.random((1, 4, 2))
    layer.add(new_shape_2)
    new_color = np.array([0, 0, 1, 1])
    expected_color = np.vstack((expected_color, new_color))
    new_properties = {'shape_type': np.array(['A', 'B'])}
    attribute_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(attribute_color, expected_color)
    np.testing.assert_equal(layer.properties, new_properties)


@pytest.mark.parametrize("attribute", ['edge', 'face'])
def test_adding_value_color_cycle(attribute):
    """Test that adding values to properties used to set a color cycle
    and then calling Shapes.refresh_colors() performs the update and adds the
    new value to the face/edge_color_cycle_map.
    """
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'shape_type': _make_cycled_properties(['A', 'B'], shape[0])}
    color_cycle = ['red', 'blue']
    shapes_kwargs = {
        'properties': properties,
        f'{attribute}_color': 'shape_type',
        f'{attribute}_color_cycle': color_cycle,
    }
    layer = Shapes(data, **shapes_kwargs)

    # make shape 0 shape_type C
    shape_types = layer.properties['shape_type']
    shape_types[0] = 'C'
    layer.properties['shape_type'] = shape_types
    layer.refresh_colors(update_color_mapping=False)

    color_cycle_map = getattr(layer, f'{attribute}_color_cycle_map')
    color_map_keys = [*color_cycle_map]
    assert 'C' in color_map_keys


@pytest.mark.parametrize("attribute", ['edge', 'face'])
def test_color_colormap(attribute):
    """Test setting edge/face color with a colormap"""
    # create Shapes using with a colormap
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'shape_type': _make_cycled_properties([0, 1.5], shape[0])}
    shapes_kwargs = {
        'properties': properties,
        f'{attribute}_color': 'shape_type',
        f'{attribute}_colormap': 'gray',
    }
    layer = Shapes(data, **shapes_kwargs)
    assert layer.properties == properties
    color_mode = getattr(layer, f'{attribute}_color_mode')
    assert color_mode == 'colormap'
    color_array = transform_color(['black', 'white'] * int(shape[0] / 2))
    attribute_color = getattr(layer, f'{attribute}_color')
    assert np.all(attribute_color == color_array)

    # change the color cycle - face_color should not change
    setattr(layer, f'{attribute}_color_cycle', ['red', 'blue'])
    attribute_color = getattr(layer, f'{attribute}_color')
    assert np.all(attribute_color == color_array)

    # Add new shape and test its color
    new_shape = np.random.random((1, 4, 2))
    layer.selected_data = {0}
    layer.add(new_shape)
    attribute_color = getattr(layer, f'{attribute}_color')
    assert len(attribute_color) == shape[0] + 1
    np.testing.assert_allclose(
        attribute_color,
        np.vstack((color_array, transform_color('black'))),
    )

    # Check removing data adjusts colors correctly
    layer.selected_data = {0, 2}
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    attribute_color = getattr(layer, f'{attribute}_color')
    assert len(attribute_color) == shape[0] - 1
    np.testing.assert_allclose(
        attribute_color,
        np.vstack(
            (
                color_array[1],
                color_array[3:],
                transform_color('black'),
            )
        ),
    )

    # adjust the clims
    setattr(layer, f'{attribute}_contrast_limits', (0, 3))
    layer.refresh_colors(update_color_mapping=False)
    attribute_color = getattr(layer, f'{attribute}_color')
    np.testing.assert_allclose(attribute_color[-2], [0.5, 0.5, 0.5, 1])

    # change the colormap
    new_colormap = 'viridis'
    setattr(layer, f'{attribute}_colormap', new_colormap)
    attribute_colormap = getattr(layer, f'{attribute}_colormap')
    assert attribute_colormap.name == new_colormap


@pytest.mark.parametrize("attribute", ['edge', 'face'])
def test_colormap_without_properties(attribute):
    """Setting the colormode to colormap should raise an exception"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)

    with pytest.raises(ValueError):
        setattr(layer, f'{attribute}_color_mode', 'colormap')


@pytest.mark.parametrize("attribute", ['edge', 'face'])
def test_colormap_with_categorical_properties(attribute):
    """Setting the colormode to colormap should raise an exception"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    properties = {'shape_type': _make_cycled_properties(['A', 'B'], shape[0])}
    layer = Shapes(data, properties=properties)

    with pytest.raises(TypeError):
        with pytest.warns(UserWarning):
            setattr(layer, f'{attribute}_color_mode', 'colormap')


@pytest.mark.parametrize("attribute", ['edge', 'face'])
def test_add_colormap(attribute):
    """Test  directly adding a vispy Colormap object"""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    annotations = {'shape_type': _make_cycled_properties([0, 1.5], shape[0])}
    color_kwarg = f'{attribute}_color'
    colormap_kwarg = f'{attribute}_colormap'
    args = {color_kwarg: 'shape_type', colormap_kwarg: 'viridis'}
    layer = Shapes(data, properties=annotations, **args)

    setattr(layer, f'{attribute}_colormap', 'gray')
    layer_colormap = getattr(layer, f'{attribute}_colormap')
    assert layer_colormap.name == 'gray'


def test_edge_width():
    """Test setting edge width."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.current_edge_width == 1
    assert len(layer.edge_width) == shape[0]
    assert layer.edge_width == [1] * shape[0]

    # With no data selected changing edge width has no effect
    layer.current_edge_width = 2
    assert layer.current_edge_width == 2
    assert layer.edge_width == [1] * shape[0]

    # Select data and change edge color of selection
    layer.selected_data = {0, 1}
    assert layer.current_edge_width == 1
    layer.current_edge_width = 3
    assert layer.edge_width == [3] * 2 + [1] * (shape[0] - 2)

    # Add new shape and test its width
    new_shape = np.random.random((1, 4, 2))
    layer.selected_data = set()
    layer.current_edge_width = 4
    layer.add(new_shape)
    assert layer.edge_width == [3] * 2 + [1] * (shape[0] - 2) + [4]

    # Instantiate with custom edge width
    layer = Shapes(data, edge_width=5)
    assert layer.current_edge_width == 5

    # Instantiate with custom edge width list
    width_list = [2, 3] * 5
    layer = Shapes(data, edge_width=width_list)
    assert layer.current_edge_width == 1
    assert layer.edge_width == width_list

    # Add new shape and test its color
    layer.current_edge_width = 4
    layer.add(new_shape)
    assert len(layer.edge_width) == shape[0] + 1
    assert layer.edge_width == width_list + [4]

    # Check removing data adjusts colors correctly
    layer.selected_data = {0, 2}
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.edge_width) == shape[0] - 1
    assert layer.edge_width == [width_list[1]] + width_list[3:] + [4]

    # Test setting edge width with number
    layer.edge_width = 4
    assert all([width == 4 for width in layer.edge_width])

    # Test setting edge width with list
    new_widths = [2] * 5 + [3] * 4
    layer.edge_width = new_widths
    assert layer.edge_width == new_widths

    # Test setting with incorrect size list throws error
    new_widths = [2, 3]
    with pytest.raises(ValueError):
        layer.edge_width = new_widths


def test_z_index():
    """Test setting z-index during instantiation."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.z_index == [0] * shape[0]

    # Instantiate with custom z-index
    layer = Shapes(data, z_index=4)
    assert layer.z_index == [4] * shape[0]

    # Instantiate with custom z-index list
    z_index_list = [2, 3] * 5
    layer = Shapes(data, z_index=z_index_list)
    assert layer.z_index == z_index_list

    # Add new shape and its z-index
    new_shape = np.random.random((1, 4, 2))
    layer.add(new_shape)
    assert len(layer.z_index) == shape[0] + 1
    assert layer.z_index == z_index_list + [4]

    # Check removing data adjusts colors correctly
    layer.selected_data = {0, 2}
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.z_index) == shape[0] - 1
    assert layer.z_index == [z_index_list[1]] + z_index_list[3:] + [4]

    # Test setting index with number
    layer.z_index = 4
    assert all([idx == 4 for idx in layer.z_index])

    # Test setting index with list
    new_z_indices = [2] * 5 + [3] * 4
    layer.z_index = new_z_indices
    assert layer.z_index == new_z_indices

    # Test setting with incorrect size list throws error
    new_z_indices = [2, 3]
    with pytest.raises(ValueError):
        layer.z_index = new_z_indices


def test_move_to_front():
    """Test moving shapes to front."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    z_index_list = [2, 3] * 5
    layer = Shapes(data, z_index=z_index_list)
    assert layer.z_index == z_index_list

    # Move selected shapes to front
    layer.selected_data = {0, 2}
    layer.move_to_front()
    assert layer.z_index == [4] + [z_index_list[1]] + [4] + z_index_list[3:]


def test_move_to_back():
    """Test moving shapes to back."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    z_index_list = [2, 3] * 5
    layer = Shapes(data, z_index=z_index_list)
    assert layer.z_index == z_index_list

    # Move selected shapes to front
    layer.selected_data = {0, 2}
    layer.move_to_back()
    assert layer.z_index == [1] + [z_index_list[1]] + [1] + z_index_list[3:]


def test_interaction_box():
    """Test the creation of the interaction box."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer._selected_box is None

    layer.selected_data = {0}
    assert len(layer._selected_box) == 10

    layer.selected_data = {0, 1}
    assert len(layer._selected_box) == 10

    layer.selected_data = set()
    assert layer._selected_box is None


def test_copy_and_paste():
    """Test copying and pasting selected shapes."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    # Clipboard starts empty
    assert layer._clipboard == {}

    # Pasting empty clipboard doesn't change data
    layer._paste_data()
    assert len(layer.data) == 10

    # Copying with nothing selected leave clipboard empty
    layer._copy_data()
    assert layer._clipboard == {}

    # Copying and pasting with two shapes selected adds to clipboard and data
    layer.selected_data = {0, 1}
    layer._copy_data()
    layer._paste_data()
    assert len(layer._clipboard) == 6
    assert len(layer.data) == shape[0] + 2
    assert np.all(
        [np.all(a == b) for a, b in zip(layer.data[:2], layer.data[-2:])]
    )

    # Pasting again adds two more shapes to data
    layer._paste_data()
    assert len(layer.data) == shape[0] + 4
    assert np.all(
        [np.all(a == b) for a, b in zip(layer.data[:2], layer.data[-2:])]
    )

    # Unselecting everything and copying and pasting will empty the clipboard
    # and add no new data
    layer.selected_data = set()
    layer._copy_data()
    layer._paste_data()
    assert layer._clipboard == {}
    assert len(layer.data) == shape[0] + 4


def test_value():
    """Test getting the value of the data at the current coordinates."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[-1, :] = [[0, 0], [0, 10], [10, 0], [10, 10]]
    layer = Shapes(data)
    value = layer.get_value((0,) * 2)
    assert value == (9, None)

    layer.mode = 'select'
    layer.selected_data = {9}
    value = layer.get_value((0,) * 2)
    assert value == (9, 7)

    layer = Shapes(data + 5)
    value = layer.get_value((0,) * 2)
    assert value == (None, None)


def test_message():
    """Test converting values and coords to message."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    msg = layer.get_status((0,) * 2)
    assert type(msg) == str


def test_thumbnail():
    """Test the image thumbnail for square data."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[-1, :] = [[0, 0], [0, 20], [20, 0], [20, 20]]
    layer = Shapes(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_to_masks():
    """Test the mask generation."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    masks = layer.to_masks()
    assert masks.ndim == 3
    assert len(masks) == shape[0]

    masks = layer.to_masks(mask_shape=[20, 20])
    assert masks.shape == (shape[0], 20, 20)


def test_to_labels():
    """Test the labels generation."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    labels = layer.to_labels()
    assert labels.ndim == 2
    assert len(np.unique(labels)) <= 11

    labels = layer.to_labels(labels_shape=[20, 20])
    assert labels.shape == (20, 20)
    assert len(np.unique(labels)) <= 11


def test_to_labels_3D():
    """Test label generation for 3D data"""
    data = [
        [[0, 100, 100], [0, 100, 200], [0, 200, 200], [0, 200, 100]],
        [[1, 125, 125], [1, 125, 175], [1, 175, 175], [1, 175, 125]],
        [[2, 100, 100], [2, 100, 200], [2, 200, 200], [2, 200, 100]],
    ]
    labels_shape = (3, 300, 300)
    layer = Shapes(np.array(data), shape_type='polygon')
    labels = layer.to_labels(labels_shape=labels_shape)
    assert np.all(labels.shape == labels_shape)
    assert np.all(np.unique(labels) == [0, 1, 2, 3])


def test_add_single_shape_consistent_properties():
    """Test adding a single shape ensures correct number of added properties"""
    data = [
        np.array([[100, 200], [200, 300]]),
        np.array([[300, 400], [400, 500]]),
    ]
    properties = {'index': [1, 2]}
    layer = Shapes(
        np.array(data), shape_type='rectangle', properties=properties
    )

    layer.add(np.array([[500, 600], [700, 800]]))
    assert len(layer.properties['index']) == 3
    assert layer.properties['index'][2] == 2


def test_add_shapes_consistent_properties():
    """Test adding multiple shapes ensures correct number of added properties"""
    data = [
        np.array([[100, 200], [200, 300]]),
        np.array([[300, 400], [400, 500]]),
    ]
    properties = {'index': [1, 2]}
    layer = Shapes(
        np.array(data), shape_type='rectangle', properties=properties
    )

    layer.add(
        [
            np.array([[500, 600], [700, 800]]),
            np.array([[700, 800], [800, 900]]),
        ]
    )
    assert len(layer.properties['index']) == 4
    assert layer.properties['index'][2] == 2
    assert layer.properties['index'][3] == 2


def test_world_data_extent():
    """Test extent after applying transforms."""
    data = [(7, -5, 0), (-2, 0, 15), (4, 30, 12)]
    layer = Shapes([data, np.add(data, [2, -3, 0])], shape_type='polygon')
    min_val = (-2, -8, 0)
    max_val = (9, 30, 15)
    extent = np.array((min_val, max_val))
    check_layer_world_data_extent(layer, extent, (3, 1, 1), (10, 20, 5))
