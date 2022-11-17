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
def test_track_layer_colorby_nonexistent():
    """Test error handling for non-existent properties with color_by"""
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


def test_malformed_graph():
    """Test for malformed graph."""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    data[50:, 0] = 1
    graph = {1: [0], 2: [33]}
    with pytest.raises(ValueError):
        Tracks(data, graph=graph)


def test_tracks_float_time_index():
    """Test Tracks layer instantiation with floating point time values"""
    coords = np.random.normal(loc=50, size=(100, 2))
    time = np.random.normal(loc=50, size=(100, 1))
    track_id = np.zeros((100, 1))
    track_id[50:] = 1
    data = np.concatenate((track_id, time, coords), axis=1)
    Tracks(data)


def test_tracks_length_change():
    """Test changing length properties of tracks"""
    track_length = 1000
    data = np.zeros((track_length, 4))
    layer = Tracks(data)
    layer.tail_length = track_length
    assert layer.tail_length == track_length
    assert layer._max_length == track_length

    layer = Tracks(data)
    layer.head_length = track_length
    assert layer.head_length == track_length
    assert layer._max_length == track_length


def test_color_by_same_after_properties_change():
    """See https://github.com/napari/napari/issues/5330"""
    data = np.array(
        [
            [1, 0, 236, 0],
            [1, 1, 236, 100],
            [1, 2, 236, 200],
            [2, 0, 436, 0],
            [2, 1, 436, 100],
            [2, 2, 436, 200],
            [3, 0, 636, 0],
            [3, 1, 636, 100],
            [3, 2, 636, 200],
        ]
    )
    np.random.seed(0)
    initial_properties = {
        'time': data[:, 1],
        'confidence': np.random.rand(data.shape[0]),
    }
    layer = Tracks(data, properties=initial_properties)
    layer.color_by = 'confidence'

    # Change the properties value by removing the time column.
    layer.properties = {'confidence': initial_properties['confidence']}

    assert layer.color_by == 'confidence'
