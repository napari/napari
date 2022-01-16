import numpy as np
import pandas as pd
import pytest
from vispy.color import get_colormap

from napari._tests.utils import check_layer_world_data_extent
from napari.layers import Vectors
from napari.utils.colormaps.standardize_color import transform_color

# Set random seed for testing
np.random.seed(0)


def test_random_vectors():
    """Test instantiating Vectors layer with random coordinate-like 2D data."""
    shape = (10, 2, 2)
    np.random.seed(0)
    data = np.random.random(shape)
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert np.all(layer.data == data)
    assert layer.data.shape == shape
    assert layer.ndim == shape[2]
    assert layer._view_data.shape[2] == 2


def test_random_vectors_image():
    """Test instantiating Vectors layer with random image-like 2D data."""
    shape = (20, 10, 2)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Vectors(data)
    assert layer.data.shape == (20 * 10, 2, 2)
    assert layer.ndim == 2
    assert layer._view_data.shape[2] == 2


def test_no_args_vectors():
    """Test instantiating Vectors layer with no arguments"""
    layer = Vectors()
    assert layer.data.shape == (0, 2, 2)


def test_no_data_vectors_with_ndim():
    """Test instantiating Vectors layers with no data but specifying ndim"""
    layer = Vectors(ndim=2)
    assert layer.data.shape[-1] == 2


def test_incompatible_ndim_vectors():
    """Test instantiating Vectors layer with ndim argument incompatible with data"""
    data = np.empty((0, 2, 2))
    with pytest.raises(ValueError):
        Vectors(data, ndim=3)


def test_empty_vectors():
    """Test instantiating Vectors layer with empty coordinate-like 2D data."""
    shape = (0, 2, 2)
    data = np.empty(shape)
    layer = Vectors(data)
    assert np.all(layer.data == data)
    assert layer.data.shape == shape
    assert layer.ndim == shape[2]
    assert layer._view_data.shape[2] == 2


def test_empty_vectors_with_property_choices():
    """Test instantiating Vectors layer with empty coordinate-like 2D data."""
    shape = (0, 2, 2)
    data = np.empty(shape)
    property_choices = {'angle': np.array([0.5], dtype=float)}
    layer = Vectors(data, property_choices=property_choices)
    assert np.all(layer.data == data)
    assert layer.data.shape == shape
    assert layer.ndim == shape[2]
    assert layer._view_data.shape[2] == 2
    np.testing.assert_equal(layer.property_choices, property_choices)


def test_empty_layer_with_edge_colormap():
    """Test creating an empty layer where the edge color is a colormap"""
    shape = (0, 2, 2)
    data = np.empty(shape)
    default_properties = {'angle': np.array([1.5], dtype=float)}
    layer = Vectors(
        data=data,
        property_choices=default_properties,
        edge_color='angle',
        edge_colormap='grays',
    )

    assert layer.edge_color_mode == 'colormap'

    # edge_color should remain empty when refreshing colors
    layer.refresh_colors(update_color_mapping=True)
    np.testing.assert_equal(layer.edge_color, np.empty((0, 4)))


def test_empty_layer_with_edge_color_cycle():
    """Test creating an empty layer where the edge color is a color cycle"""
    shape = (0, 2, 2)
    data = np.empty(shape)
    default_properties = {'vector_type': np.array(['A'])}
    layer = Vectors(
        data=data,
        property_choices=default_properties,
        edge_color='vector_type',
    )

    assert layer.edge_color_mode == 'cycle'

    # edge_color should remain empty when refreshing colors
    layer.refresh_colors(update_color_mapping=True)
    np.testing.assert_equal(layer.edge_color, np.empty((0, 4)))


def test_random_3D_vectors():
    """Test instantiating Vectors layer with random coordinate-like 3D data."""
    shape = (10, 2, 3)
    np.random.seed(0)
    data = np.random.random(shape)
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert np.all(layer.data == data)
    assert layer.data.shape == shape
    assert layer.ndim == shape[2]
    assert layer._view_data.shape[2] == 2


def test_random_3D_vectors_image():
    """Test instantiating Vectors layer with random image-like 3D data."""
    shape = (12, 20, 10, 3)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Vectors(data)
    assert layer.data.shape == (12 * 20 * 10, 2, 3)
    assert layer.ndim == 3
    assert layer._view_data.shape[2] == 2


def test_no_data_3D_vectors_with_ndim():
    """Test instantiating Vectors layers with no data but specifying ndim"""
    layer = Vectors(ndim=3)
    assert layer.data.shape[-1] == 3


@pytest.mark.filterwarnings("ignore:Passing `np.nan`:DeprecationWarning:numpy")
def test_empty_3D_vectors():
    """Test instantiating Vectors layer with empty coordinate-like 3D data."""
    shape = (0, 2, 3)
    data = np.empty(shape)
    layer = Vectors(data)
    assert np.all(layer.data == data)
    assert layer.data.shape == shape
    assert layer.ndim == shape[2]
    assert layer._view_data.shape[2] == 2


def test_data_setter():
    n_vectors_0 = 10
    shape = (n_vectors_0, 2, 3)
    np.random.seed(0)
    data = np.random.random(shape)
    data[:, 0, :] = 20 * data[:, 0, :]
    properties = {
        'prop_0': np.random.random((n_vectors_0,)),
        'prop_1': np.random.random((n_vectors_0,)),
    }
    layer = Vectors(data, properties=properties)

    assert len(layer.data) == n_vectors_0
    assert len(layer.edge_color) == n_vectors_0
    assert len(layer.properties['prop_0']) == n_vectors_0
    assert len(layer.properties['prop_1']) == n_vectors_0

    # set the data with more vectors
    n_vectors_1 = 20
    data_1 = np.random.random((n_vectors_1, 2, 3))
    data_1[:, 0, :] = 20 * data_1[:, 0, :]
    layer.data = data_1

    assert len(layer.data) == n_vectors_1
    assert len(layer.edge_color) == n_vectors_1
    assert len(layer.properties['prop_0']) == n_vectors_1
    assert len(layer.properties['prop_1']) == n_vectors_1

    # set the data with fewer vectors
    n_vectors_2 = 5
    data_2 = np.random.random((n_vectors_2, 2, 3))
    data_2[:, 0, :] = 20 * data_2[:, 0, :]
    layer.data = data_2

    assert len(layer.data) == n_vectors_2
    assert len(layer.edge_color) == n_vectors_2
    assert len(layer.properties['prop_0']) == n_vectors_2
    assert len(layer.properties['prop_1']) == n_vectors_2


def test_properties_dataframe():
    """test if properties can be provided as a DataFrame"""
    shape = (10, 2)
    np.random.seed(0)
    shape = (10, 2, 2)
    data = np.random.random(shape)
    data[:, 0, :] = 20 * data[:, 0, :]
    properties = {'vector_type': np.array(['A', 'B'] * int(shape[0] / 2))}
    properties_df = pd.DataFrame(properties)
    properties_df = properties_df.astype(properties['vector_type'].dtype)
    layer = Vectors(data, properties=properties_df)
    np.testing.assert_equal(layer.properties, properties)

    # test adding a dataframe via the properties setter
    properties_2 = {'vector_type2': np.array(['A', 'B'] * int(shape[0] / 2))}
    properties_df2 = pd.DataFrame(properties_2)
    layer.properties = properties_df2
    np.testing.assert_equal(layer.properties, properties_2)


def test_adding_properties():
    """test adding properties to a Vectors layer"""
    shape = (10, 2)
    np.random.seed(0)
    shape = (10, 2, 2)
    data = np.random.random(shape)
    data[:, 0, :] = 20 * data[:, 0, :]
    properties = {'vector_type': np.array(['A', 'B'] * int(shape[0] / 2))}
    layer = Vectors(data)

    # properties should start empty
    assert layer.properties == {}

    # add properties
    layer.properties = properties
    np.testing.assert_equal(layer.properties, properties)

    # removing a property that was the _edge_color_property should give a warning
    layer.edge_color = 'vector_type'
    properties_2 = {
        'not_vector_type': np.array(['A', 'B'] * int(shape[0] / 2))
    }
    with pytest.warns(RuntimeWarning):
        layer.properties = properties_2

    # adding properties with the wrong length should raise an exception
    bad_properties = {'vector_type': np.array(['A', 'B'])}
    with pytest.raises(ValueError):
        layer.properties = bad_properties


def test_changing_data():
    """Test changing Vectors data."""
    shape_a = (10, 2, 2)
    np.random.seed(0)
    data_a = np.random.random(shape_a)
    data_a[:, 0, :] = 20 * data_a[:, 0, :]
    shape_b = (16, 2, 2)
    data_b = np.random.random(shape_b)
    data_b[:, 0, :] = 20 * data_b[:, 0, :]
    layer = Vectors(data_b)
    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.data.shape == shape_b
    assert layer.ndim == shape_b[2]
    assert layer._view_data.shape[2] == 2


def test_name():
    """Test setting layer name."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert layer.name == 'Vectors'

    layer = Vectors(data, name='random')
    assert layer.name == 'random'

    layer.name = 'vcts'
    assert layer.name == 'vcts'


def test_visiblity():
    """Test setting layer visibility."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Vectors(data, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_opacity():
    """Test setting layer opacity."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert layer.opacity == 0.7

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer = Vectors(data, opacity=0.6)
    assert layer.opacity == 0.6

    layer.opacity = 0.3
    assert layer.opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Vectors(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


def test_edge_width():
    """Test setting edge width."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert layer.edge_width == 1

    layer.edge_width = 2
    assert layer.edge_width == 2

    layer = Vectors(data, edge_width=3)
    assert layer.edge_width == 3


def test_invalid_edge_color():
    """Test providing an invalid edge color raises an exception"""
    np.random.seed(0)
    shape = (10, 2, 2)
    data = np.random.random(shape)
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)

    with pytest.raises(ValueError):
        layer.edge_color = 5


def test_edge_color_direct():
    """Test setting edge color."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    np.testing.assert_allclose(
        layer.edge_color, np.repeat([[1, 0, 0, 1]], data.shape[0], axis=0)
    )

    # set edge color as an RGB array
    layer.edge_color = [0, 0, 1]
    np.testing.assert_allclose(
        layer.edge_color, np.repeat([[0, 0, 1, 1]], data.shape[0], axis=0)
    )

    # set edge color as an RGBA array
    layer.edge_color = [0, 1, 0, 0.5]
    np.testing.assert_allclose(
        layer.edge_color, np.repeat([[0, 1, 0, 0.5]], data.shape[0], axis=0)
    )

    # set all edge colors directly
    edge_colors = np.random.random((data.shape[0], 4))
    layer.edge_color = edge_colors
    np.testing.assert_allclose(layer.edge_color, edge_colors)


def test_edge_color_cycle():
    """Test creating Vectors where edge color is set by a color cycle"""
    np.random.seed(0)
    shape = (10, 2, 2)
    data = np.random.random(shape)
    data[:, 0, :] = 20 * data[:, 0, :]
    properties = {'vector_type': np.array(['A', 'B'] * int(shape[0] / 2))}
    color_cycle = ['red', 'blue']
    layer = Vectors(
        data,
        properties=properties,
        edge_color='vector_type',
        edge_color_cycle=color_cycle,
    )
    np.testing.assert_equal(layer.properties, properties)
    edge_color_array = transform_color(color_cycle * int(shape[0] / 2))
    assert np.all(layer.edge_color == edge_color_array)


def test_edge_color_colormap():
    """Test creating Vectors where edge color is set by a colormap"""
    shape = (10, 2)
    shape = (10, 2, 2)
    data = np.random.random(shape)
    data[:, 0, :] = 20 * data[:, 0, :]
    properties = {'angle': np.array([0, 1.5] * int(shape[0] / 2))}
    layer = Vectors(
        data,
        properties=properties,
        edge_color='angle',
        edge_colormap='gray',
    )
    np.testing.assert_equal(layer.properties, properties)
    assert layer.edge_color_mode == 'colormap'
    edge_color_array = transform_color(['black', 'white'] * int(shape[0] / 2))
    assert np.all(layer.edge_color == edge_color_array)

    # change the color cycle - edge_color should not change
    layer.edge_color_cycle = ['red', 'blue']
    assert np.all(layer.edge_color == edge_color_array)

    # adjust the clims
    layer.edge_contrast_limits = (0, 3)
    layer.refresh_colors(update_color_mapping=False)
    np.testing.assert_allclose(layer.edge_color[-1], [0.5, 0.5, 0.5, 1])

    # change the colormap
    new_colormap = 'viridis'
    layer.edge_colormap = new_colormap
    assert layer.edge_colormap.name == new_colormap

    # test adding a colormap with a vispy Colormap object
    layer.edge_colormap = get_colormap('gray')
    assert 'unnamed colormap' in layer.edge_colormap.name


def test_edge_color_map_non_numeric_property():
    """Test setting edge_color as a color map of a
    non-numeric property raises an error
    """
    np.random.seed(0)
    shape = (10, 2, 2)
    data = np.random.random(shape)
    data[:, 0, :] = 20 * data[:, 0, :]
    properties = {'vector_type': np.array(['A', 'B'] * int(shape[0] / 2))}
    color_cycle = ['red', 'blue']
    initial_color = [0, 1, 0, 1]
    layer = Vectors(
        data,
        properties=properties,
        edge_color=initial_color,
        edge_color_cycle=color_cycle,
        edge_colormap='gray',
    )
    # layer should start out in direct edge color mode with all green vectors
    assert layer.edge_color_mode == 'direct'
    np.testing.assert_allclose(
        layer.edge_color, np.repeat([initial_color], shape[0], axis=0)
    )

    # switching to colormap mode should raise an error because the 'vector_type' is non-numeric
    layer.edge_color = 'vector_type'
    with pytest.raises(TypeError):
        layer.edge_color_mode = 'colormap'


@pytest.mark.filterwarnings("ignore:elementwise comparis:FutureWarning:numpy")
def test_switching_edge_color_mode():
    """Test transitioning between all color modes"""
    np.random.seed(0)
    shape = (10, 2, 2)
    data = np.random.random(shape)
    data[:, 0, :] = 20 * data[:, 0, :]
    properties = {
        'magnitude': np.arange(shape[0]),
        'vector_type': np.array(['A', 'B'] * int(shape[0] / 2)),
    }
    color_cycle = ['red', 'blue']
    initial_color = [0, 1, 0, 1]
    layer = Vectors(
        data,
        properties=properties,
        edge_color=initial_color,
        edge_color_cycle=color_cycle,
        edge_colormap='gray',
    )
    # layer should start out in direct edge color mode with all green vectors
    assert layer.edge_color_mode == 'direct'
    np.testing.assert_allclose(
        layer.edge_color, np.repeat([initial_color], shape[0], axis=0)
    )

    # there should not be an edge_color_property
    assert layer._edge.color_properties is None

    # transitioning to colormap should raise a warning
    # because there isn't an edge color property yet and
    # the first property in Vectors.properties is being automatically selected
    with pytest.warns(RuntimeWarning):
        layer.edge_color_mode = 'colormap'
    assert layer._edge.color_properties.name == next(iter(properties))
    np.testing.assert_allclose(layer.edge_color[-1], [1, 1, 1, 1])

    # switch to color cycle
    layer.edge_color_mode = 'cycle'
    layer.edge_color = 'vector_type'
    edge_color_array = transform_color(color_cycle * int(shape[0] / 2))
    np.testing.assert_allclose(layer.edge_color, edge_color_array)

    # switch back to direct, edge_colors shouldn't change
    edge_colors = layer.edge_color
    layer.edge_color_mode = 'direct'
    np.testing.assert_allclose(layer.edge_color, edge_colors)


def test_properties_color_mode_without_properties():
    """Test that switching to a colormode requiring
    properties without properties defined raises an exceptions
    """
    np.random.seed(0)
    shape = (10, 2, 2)
    data = np.random.random(shape)
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert layer.properties == {}

    with pytest.raises(ValueError):
        layer.edge_color_mode = 'colormap'

    with pytest.raises(ValueError):
        layer.edge_color_mode = 'cycle'


def test_length():
    """Test setting length."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    assert layer.length == 1

    layer.length = 2
    assert layer.length == 2

    layer = Vectors(data, length=3)
    assert layer.length == 3


def test_thumbnail():
    """Test the image thumbnail for square data."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 18 * data[:, 0, :] + 1
    data[0, :, :] = [0, 0]
    data[-1, 0, :] = [20, 20]
    data[-1, 1, :] = [0, 0]
    layer = Vectors(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_big_thumbail():
    """Test the image thumbnail with n_vectors > _max_vectors_thumbnail"""
    np.random.seed(0)
    n_vectors = int(1.5 * Vectors._max_vectors_thumbnail)
    data = np.random.random((n_vectors, 2, 2))
    data[:, 0, :] = 18 * data[:, 0, :] + 1
    data[0, :, :] = [0, 0]
    data[-1, 0, :] = [20, 20]
    data[-1, 1, :] = [0, 0]
    layer = Vectors(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_value():
    """Test getting the value of the data at the current coordinates."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    value = layer.get_value((0,) * 2)
    assert value is None


@pytest.mark.parametrize(
    'position,view_direction,dims_displayed,world',
    [
        ((0, 0, 0), [1, 0, 0], [0, 1, 2], False),
        ((0, 0, 0), [1, 0, 0], [0, 1, 2], True),
        ((0, 0, 0, 0), [0, 1, 0, 0], [1, 2, 3], True),
    ],
)
def test_value_3d(position, view_direction, dims_displayed, world):
    """Currently get_value should return None in 3D"""
    np.random.seed(0)
    data = np.random.random((10, 2, 3))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    layer._slice_dims([0, 0, 0], ndisplay=3)
    value = layer.get_value(
        position,
        view_direction=view_direction,
        dims_displayed=dims_displayed,
        world=world,
    )
    assert value is None


def test_message():
    """Test converting value and coords to message."""
    np.random.seed(0)
    data = np.random.random((10, 2, 2))
    data[:, 0, :] = 20 * data[:, 0, :]
    layer = Vectors(data)
    msg = layer.get_status((0,) * 2)
    assert type(msg) == str


def test_world_data_extent():
    """Test extent after applying transforms."""
    # data input format is start position, then length.
    data = [[(7, -5, -3), (1, -1, 2)], [(0, 0, 0), (4, 30, 12)]]
    min_val = (0, -6, -3)
    max_val = (8, 30, 12)
    layer = Vectors(np.array(data))
    extent = np.array((min_val, max_val))
    check_layer_world_data_extent(layer, extent, (3, 1, 1), (10, 20, 5), False)
