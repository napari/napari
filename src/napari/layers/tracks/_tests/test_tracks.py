import numpy as np
import pandas as pd
import pytest

from napari.components.dims import Dims
from napari.layers import Tracks
from napari.layers.tracks._track_utils import TrackManager
from napari.utils._test_utils import (
    validate_all_params_in_docstring,
    validate_kwargs_sorted,
)

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
    'data', [data_array_2dt, data_list_2dt, dataframe_2dt]
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
    'data', [data_array_3dt, data_list_3dt, dataframe_3dt]
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
    np.testing.assert_array_equal(layer.data, data)


@pytest.mark.parametrize(
    'timestamps', [np.arange(100, 200), np.arange(100, 300, 2)]
)
def test_track_layer_data_nonzero_starting_time(timestamps):
    """Test data with sparse timestamps or not starting at zero."""
    data = np.zeros((100, 4))
    data[:, 1] = timestamps
    layer = Tracks(data)
    np.testing.assert_array_equal(layer.data, data)


def test_track_layer_data_flipped():
    """Test data flipped."""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    data[:, 0] = np.arange(100)
    data = np.flip(data, axis=0)
    layer = Tracks(data)
    np.testing.assert_array_equal(layer.data, np.flip(data, axis=0))


properties_dict = {'time': np.arange(100)}
properties_df = pd.DataFrame(properties_dict)


@pytest.mark.parametrize('properties', [{}, properties_dict, properties_df])
def test_track_layer_properties(properties):
    """Test properties."""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    layer = Tracks(data, properties=properties)
    for k, v in properties.items():
        np.testing.assert_equal(layer.properties[k], v)


@pytest.mark.parametrize('properties', [{}, properties_dict, properties_df])
def test_track_layer_properties_flipped(properties):
    """Test properties."""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    data[:, 0] = np.arange(100)
    data = np.flip(data, axis=0)
    layer = Tracks(data, properties=properties)
    for k, v in properties.items():
        np.testing.assert_equal(layer.properties[k], np.flip(v))


@pytest.mark.filterwarnings('ignore:.*track_id.*:UserWarning')
def test_track_layer_colorby_nonexistent():
    """Test error handling for non-existent properties with color_by"""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    non_existant_property = 'not_a_valid_key'
    assert non_existant_property not in properties_dict
    with pytest.raises(ValueError, match='not a valid property'):
        Tracks(
            data, properties=properties_dict, color_by=non_existant_property
        )


def test_track_layer_properties_changed_colorby():
    """Test behaviour when changes to properties invalidate current color_by"""
    properties_dict_1 = {'time': np.arange(100), 'prop1': np.arange(100)}
    properties_dict_2 = {'time': np.arange(100), 'prop2': np.arange(100)}
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    layer = Tracks(data, properties=properties_dict_1, color_by='prop1')
    # test warning is raised
    with pytest.warns(UserWarning, match='Falling back to track_id'):
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
    np.testing.assert_array_equal(layer.data, cropped_data)
    assert layer.graph == {}


def test_malformed_id():
    """Test for malformed track ID."""
    data = np.random.random((100, 4))
    data[:, 1] = np.arange(100)
    with pytest.raises(ValueError, match='must be an integer'):
        Tracks(data)


def test_malformed_graph():
    """Test for malformed graph."""
    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    data[50:, 0] = 1
    graph = {1: [0], 2: [33]}
    with pytest.raises(ValueError, match='node 2 not found'):
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


def test_fast_points_lookup() -> None:
    # creates sorted points
    time_points = np.asarray([0, 1, 3, 5, 10])
    repeats = np.asarray([3, 4, 6, 3, 5])
    sorted_time = np.repeat(time_points, repeats)
    end = np.cumsum(repeats)
    start = np.insert(end[:-1], 0, 0)

    # compute lookup
    points_lookup = TrackManager._fast_points_lookup(sorted_time)

    assert len(time_points) == len(points_lookup)
    total_length = 0
    for s, e, t, r in zip(start, end, time_points, repeats, strict=False):
        assert points_lookup[t].start == s
        assert points_lookup[t].stop == e
        assert points_lookup[t].stop - points_lookup[t].start == r
        unique_time = sorted_time[points_lookup[t]]
        np.testing.assert_array_equal(unique_time[0], unique_time)
        total_length += len(unique_time)

    assert total_length == len(sorted_time)


def test_single_time_tracks() -> None:
    """Edge case where all tracks belong to a single time"""

    # track_id, t, y, x
    tracks = [[0, 5, 2, 3], [1, 5, 3, 4], [2, 5, 4, 5]]
    layer = Tracks(tracks)

    np.testing.assert_array_equal(layer.data, tracks)


def test_track_ids_ordering() -> None:
    """Check if tracks ids are correctly set to features when given not-sorted tracks."""
    # track_id, t, y, x
    unsorted_data = np.asarray(
        [[1, 1, 0, 0], [0, 1, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]
    )
    sorted_track_ids = [0, 0, 1, 1, 2]  # track_ids after sorting

    layer = Tracks(unsorted_data)
    np.testing.assert_array_equal(sorted_track_ids, layer.features['track_id'])


def test_changing_data_inplace() -> None:
    """Test if layer can be refreshed after changing data in place."""

    data = np.ones((100, 4))
    data[:, 1] = np.arange(100)

    layer = Tracks(data)

    # Change data in place
    # coordinates
    layer.data[50:, -1] = 2
    layer.refresh()

    # time
    layer.data[50:, 1] = np.arange(100, 150)
    layer.refresh()

    # track_id
    layer.data[50:, 0] = 2
    layer.refresh()


def test_track_connex_validity() -> None:
    """Test if track_connex is valid (i.e if the value False appears as many
    times as there are tracks."""

    data = np.zeros((11, 4))

    # Track ids
    data[:-1, 0] = np.repeat(np.arange(1, 6), 2)
    # create edge case where a track has length one
    data[-1, 0] = 6

    # Time
    data[:-1, 1] = np.array([0, 1] * 5)
    data[-1, 1] = 0

    layer = Tracks(data)

    # number of tracks
    n_tracks = 6

    # the number of 'False' in the track_connex array should be equal to the number of tracks
    assert np.sum(~layer._manager.track_connex) == n_tracks


def test_track_coloring() -> None:
    """Test if the track colors are correctly set."""

    data = np.zeros((100, 4))
    data[:, 1] = np.arange(100)
    layer = Tracks(data)

    colors = np.random.random(size=(100, 4))
    layer.track_colors = colors

    assert np.array_equal(layer._track_colors, colors)


def test_hide_completed_tracks() -> None:
    """Test that hide_completed_tracks correctly masks track_connex for completed tracks."""

    # Create test data with multiple tracks that finish at different times
    # Track 0: time 0-2 (finishes at t=2)
    # Track 1: time 3-5 (finishes at t=5)
    # Track 2: time 0-7 (finishes at t=7)
    data = np.array(
        [
            [0, 0, 10, 10],  # Track 0
            [0, 1, 11, 11],
            [0, 2, 12, 12],
            [1, 3, 20, 20],  # Track 1
            [1, 4, 21, 21],
            [1, 5, 22, 22],
            [2, 0, 30, 30],  # Track 2
            [2, 1, 31, 31],
            [2, 2, 32, 32],
            [2, 3, 33, 33],
            [2, 4, 34, 34],
            [2, 5, 35, 35],
            [2, 6, 36, 36],
            [2, 7, 37, 37],
        ]
    )

    layer = Tracks(data)

    # Test initial state - hide_completed_tracks should be False by default
    assert layer.hide_completed_tracks is False

    # Get original track_connex (should not be masked)
    original_connex = layer.track_connex.copy()

    # Enable hide_completed_tracks
    layer.hide_completed_tracks = True

    # Test at time point 4 (Track 0 completed at t=2, Track 1 and 2 still active)
    # Need to explicitly set range to include time 4
    dims = Dims(
        ndim=layer.ndim,
        point=(4, 0, 0),
        range=((0, 8, 1), (0, 50, 1), (0, 50, 1)),
    )  # Extended range for time
    layer._slice_dims(dims)

    masked_connex = layer.track_connex

    # Track 0 should be completely masked (completed before t=4)
    track_0_indices = np.where(layer.data[:, 0] == 0)[0]

    # All connections for track 0 should be False when hide_completed_tracks is enabled
    assert np.all(~masked_connex[track_0_indices])  # all False

    # Track 1 and 2 should still have their original connections (not completed)
    track_1_indices = np.where(layer.data[:, 0] == 1)[0]
    track_2_indices = np.where(layer.data[:, 0] == 2)[0]

    # For tracks 1 and 2, connections should match original (except last vertex of each track)
    np.testing.assert_array_equal(
        masked_connex[track_1_indices], original_connex[track_1_indices]
    )
    np.testing.assert_array_equal(
        masked_connex[track_2_indices], original_connex[track_2_indices]
    )

    # Test at time point 6 (Track 0 and 1 completed, Track 2 still active)
    dims = Dims(
        ndim=layer.ndim,
        point=(6, 0, 0),
        range=((0, 8, 1), (0, 50, 1), (0, 50, 1)),
    )
    layer._slice_dims(dims)
    masked_connex_6 = layer.track_connex

    # Track 0 and 1 should be masked
    assert np.all(~masked_connex_6[track_0_indices])  # all False
    assert np.all(~masked_connex_6[track_1_indices])  # all False

    # Track 2 should still have original connections
    np.testing.assert_array_equal(
        masked_connex_6[track_2_indices], original_connex[track_2_indices]
    )

    # Test at time point 8 (all tracks completed)
    dims = Dims(
        ndim=layer.ndim,
        point=(8, 0, 0),
        range=((0, 10, 1), (0, 50, 1), (0, 50, 1)),
    )
    layer._slice_dims(dims)
    all_masked_connex = layer.track_connex

    # All tracks should be masked
    assert np.all(~all_masked_connex)  # all False

    # Test disabling hide_completed_tracks
    layer.hide_completed_tracks = False
    unmasked_connex = layer.track_connex

    # Should return to original state
    np.testing.assert_array_equal(unmasked_connex, original_connex)


def test_docstring():
    validate_all_params_in_docstring(Tracks)
    validate_kwargs_sorted(Tracks)
