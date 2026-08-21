"""
Prapare Tracks example
=====================

This example demonstrates how to create a Tracks layer from a pandas
DataFrame by mapping column names to the coordinate and track id
fields expected by napari. The helper function prepare_tracks_data
is used to convert the DataFrame into the track array format required by
viewer.add_tracks.

The example shows how custom column names can be mapped to track ID,
time, and spatial coordinates before visualization.

.. tags:: tracks, dataframe, pandas, visualization

"""

import pandas as pd

import napari
from napari.layers.tracks._track_utils import prepare_tracks_data

df = pd.DataFrame(
    {
        'particle': [0, 0, 0, 1, 1, 1],
        'frame': [0, 1, 2, 0, 1, 2],
        'row': [10, 15, 20, 50, 55, 60],
        'col': [10, 15, 20, 60, 65, 70],
    }
)

column_map = {
    'track_id': 'particle',
    't': 'frame',
    'y': 'row',
    'x': 'col',
}

tracks = prepare_tracks_data(df, column_map)

viewer = napari.Viewer()
viewer.add_tracks(tracks, name="cell tracks")

napari.run()
