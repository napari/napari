import numpy as np
import pytest

from napari.layers import Tracks

# def test_empty_tracks():
#     """Test instantiating Tracks layer without data."""
#     pts = Tracks()
#     assert pts.data.shape == (0, 4)


def test_tracks_layer_2dt_ndim():
    """Test instantiating Tracks layer, check 2D+t dimensionality."""
    data = np.zeros((1, 4))
    layer = Tracks(data)
    assert layer.ndim == 3


def test_tracks_layer_3dt_ndim():
    """Test instantiating Tracks layer, check 3D+t dimensionality."""
    data = np.zeros((1, 5))
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


def test_track_layer_properties():
    """Test properties."""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    properties = {'time': data[:, 1]}
    layer = Tracks(data, properties=properties)
    assert layer.properties == properties


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
