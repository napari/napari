import numpy as np
from numpy.testing import assert_array_equal

from napari.layers.tracks._track_utils import prepare_tracks_data


def test_prepare_tracks_data():
    import pandas as pd

    # without z
    df = pd.DataFrame(
        {
            'particle': [1],
            'frame': [10],
            'row': [100],
            'col': [300],
        }
    )

    tracks = prepare_tracks_data(
        df,
        track_id='particle',
        t='frame',
        y='row',
        x='col',
    )

    expected = np.array([[1, 10, 100, 300]])

    assert_array_equal(tracks, expected)
    assert tracks.shape == (1, 4)

    # with z
    df = pd.DataFrame(
        {
            'particle': [1],
            'frame': [10],
            'depth': [5],
            'row': [100],
            'col': [300],
        }
    )

    tracks = prepare_tracks_data(
        df,
        track_id='particle',
        t='frame',
        z='depth',
        y='row',
        x='col',
    )

    expected = np.array([[1, 10, 5, 100, 300]])

    assert_array_equal(tracks, expected)
    assert tracks.shape == (1, 5)
