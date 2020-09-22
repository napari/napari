import numpy as np

from napari.layers import Tracks


def test_tracks_layer_2dt_ndim():
    """Test instantiating Tracks layer, check 2D+t dimensionality."""
    data = np.zeros((1, 3))
    properties = {'track_id': [0]}
    layer = Tracks(data, properties=properties)
    assert layer.ndim == 3


def test_tracks_layer_3dt_ndim():
    """Test instantiating Tracks layer, check 3D+t dimensionality."""
    data = np.zeros((1, 4))
    properties = {'track_id': [0]}
    layer = Tracks(data, properties=properties)
    assert layer.ndim == 4


# def test_2dt_tracks():
#     """Test instantiating Tracks layer with 2D+t data."""
#
#     data, properties, graph = tracks_2d()
#     layer = Tracks(data, properties=properties)
#
#     assert layer.ndim == 3
#     assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
#     assert np.all(
#         [np.all(lp == p) for lp, p in zip(layer.properties, properties)]
#     )
#
#
# def test_3dt_tracks():
#     """Test instantiating Tracks layer with 3D+t data."""
#
#     data, properties, graph = tracks_3d()
#     layer = Tracks(data, properties=properties)
#
#     assert layer.ndim == 4
#     assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
#     assert np.all(
#         [np.all(lp == p) for lp, p in zip(layer.properties, properties)]
#     )
