import numpy as np
import pandas as pd
import pytest

from napari.layers import Tracks

# def test_empty_tracks():
#     """Test instantiating Tracks layer without data."""
#     pts = Tracks()
#     assert pts.data.shape == (0, 4)


data_array_2dt = np.zeros((1, 4))
data_list_2dt = list(data_array_2dt)
dataframe_2dt = pd.DataFrame(
    data=data_array_2dt, columns=['track_id', 't', 'y', 'x']
)


@pytest.mark.parametrize(
    "data", [data_array_2dt, data_list_2dt, dataframe_2dt]
)
def test_tracks_layer_2dt_ndim(data):
    """Test instantiating Tracks layer, check 2D+t dimensionality."""
    layer = Tracks(data)
    assert layer.ndim == 3


data_array_3dt = np.zeros((1, 5))
data_list_3dt = list(data_array_3dt)
dataframe_3dt = pd.DataFrame(
    data=data_array_3dt, columns=['track_id', 't', 'z', 'y', 'x']
)


@pytest.mark.parametrize(
    "data", [data_array_3dt, data_list_3dt, dataframe_3dt]
)
def test_tracks_layer_3dt_ndim(data):
    """Test instantiating Tracks layer, check 3D+t dimensionality."""
    layer = Tracks(data)
    assert layer.ndim == 4


def test_track_layer_name():
    """Test track name."""
    data = np.zeros((1, 4))
    layer = Tracks(data, name='test_tracks')
    assert layer.name == 'test_tracks'


def test_track_layer_data():
    """Test data."""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    layer = Tracks(data)
    assert np.all(layer.data == data)


@pytest.mark.parametrize(
    "timestamps", [np.arange(100, 200), np.arange(100, 300, 2)]
)
def test_track_layer_data_nonzero_starting_time(timestamps):
    """Test data with sparse timestamps or not starting at zero."""
    data = np.zeros((100, 4))
    data[:, 1] = timestamps
    layer = Tracks(data)
    assert np.all(layer.data == data)


def test_track_layer_data_flipped():
    """Test data flipped."""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    data[:, 0] = np.arange(100)
    data = np.flip(data, axis=0)
    layer = Tracks(data)
    assert np.all(layer.data == np.flip(data, axis=0))


properties_dict = {'time': np.arange(100)}
properties_df = pd.DataFrame(properties_dict)


@pytest.mark.parametrize("properties", [{}, properties_dict, properties_df])
def test_track_layer_properties(properties):
    """Test properties."""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    layer = Tracks(data, properties=properties)
    for k, v in properties.items():
        np.testing.assert_equal(layer.properties[k], v)


@pytest.mark.parametrize("properties", [{}, properties_dict, properties_df])
def test_track_layer_properties_flipped(properties):
    """Test properties."""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    data[:, 0] = np.arange(100)
    data = np.flip(data, axis=0)
    layer = Tracks(data, properties=properties)
    for k, v in properties.items():
        np.testing.assert_equal(layer.properties[k], np.flip(v))


@pytest.mark.filterwarnings("ignore:.*track_id.*:UserWarning")
def test_track_layer_colorby_nonexistant():
    """Test error handling for non-existant properties with color_by"""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    non_existant_property = 'not_a_valid_key'
    assert non_existant_property not in properties_dict.keys()
    with pytest.raises(ValueError):
        Tracks(
            data, properties=properties_dict, color_by=non_existant_property
        )


@pytest.mark.filterwarnings("ignore:.*track_id.*:UserWarning")
def test_track_layer_properties_changed_colorby():
    """Test behaviour when changes to properties invalidate current color_by"""
    properties_dict_1 = {'time': np.arange(100), 'prop1': np.arange(100)}
    properties_dict_2 = {'time': np.arange(100), 'prop2': np.arange(100)}
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    layer = Tracks(data, properties=properties_dict_1, color_by='prop1')
    # test warning is raised
    with pytest.warns(UserWarning):
        layer.properties = properties_dict_2
    # test default fallback
    assert layer.color_by == 'track_id'


def test_track_layer_graph():
    """Test track layer graph."""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    data[50:, 0] = 1
    graph = {1: [0]}
    layer = Tracks(data, graph=graph)
    assert layer.graph == graph


def test_track_layer_reset_data():
    """Test changing data once layer is instantiated."""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    data[50:, 0] = 1
    properties = {'time': data[:, 1]}
    graph = {1: [0]}
    layer = Tracks(data, graph=graph, properties=properties)
    cropped_data = data[:10, :]
    layer.data = cropped_data
    assert np.all(layer.data == cropped_data)
    assert layer.graph == {}


def test_malformed_id():
    """Test for malformed track ID."""
    data = np.random.random((100, 4))
    data[:, 1] = np.arange(100)
    with pytest.raises(ValueError):
        Tracks(data)


def test_malformed_timestamps():
    """Test for malformed track timestamps."""
    data = np.random.random((100, 4))
    data[:, 0] = 0
    with pytest.raises(ValueError):
        Tracks(data)


def test_malformed_graph():
    """Test for malformed graph."""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    data[50:, 0] = 1
    graph = {1: [0], 2: [33]}
    with pytest.raises(ValueError):
        Tracks(data, graph=graph)
