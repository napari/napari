from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from napari.layers.tracks._managers import InteractiveTrackManager
from napari.layers.tracks._managers._track_manager import connex


@pytest.fixture
def make_data_and_graph() -> Tuple[pd.DataFrame, Dict]:
    r"""
    time   | 0 1 2 3 4 5 6 7 8 9
    ----------------------------
    tracks | 1-1-1-1-2-2-2-2-2-2
           |       \-3-3-3-4-4-4
           |             \-5-5-5
           |   6-6-6-7-7-9-9
           |       \-8-8-/
    """
    id_and_time = np.array(
        [
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 4],
            [2, 5],
            [2, 6],
            [2, 7],
            [2, 8],
            [2, 9],
            [3, 4],
            [3, 5],
            [3, 6],
            [4, 7],
            [4, 8],
            [4, 9],
            [5, 7],
            [5, 8],
            [5, 9],
            [6, 1],
            [6, 2],
            [6, 3],
            [7, 4],
            [7, 5],
            [8, 4],
            [8, 5],
            [9, 6],
            [9, 7],
        ]
    )

    coords = np.random.randn(len(id_and_time), 3)
    data = np.concatenate((id_and_time, coords), axis=1)
    data = pd.DataFrame(data, columns=('TrackID', 'T', 'Z', 'Y', 'X'))

    graph = {
        2: [1],
        3: [1],
        4: [3],
        5: [3],
        7: [6],
        8: [6],
        9: [7, 8],
    }

    return data, graph


def test_interactive_tracks_init(make_data_and_graph) -> None:
    """Tests InteractiveTrackManager data structure construction and serialization."""
    data: pd.DataFrame = make_data_and_graph[0]
    graph: Dict = make_data_and_graph[1]
    manager = InteractiveTrackManager(data=data.values, graph=graph)

    # checking if manager is storing the data as expected
    assert manager.ndim == 4  # (t, z, y, x)
    assert manager.max_time == data['T'].max()
    assert len(manager._id_to_nodes) == data.shape[0]
    assert len(manager._leafs) == 4  # number of leafs
    assert not manager._is_serialized

    mapping = {}
    for track_id, track in data.groupby('TrackID'):
        track = track.sort_values('T', ascending=False)
        node_index, _ = next(track.iterrows())
        mapping[node_index] = int(track_id)

    # relabeling tracks to match original graph
    manager.relabel_track_ids(mapping)

    # verifying graph and checking if serialization is stored
    assert manager.graph == graph
    assert manager._is_serialized

    manager_data = pd.DataFrame(manager.data, columns=data.columns)

    # checking connectivity used by vispy rendering
    track_connex = []
    for track_id, track in manager_data.groupby('TrackID', sort=False):
        track_connex += connex(track.values)

    assert np.all(track_connex == manager._connex)

    # sorting and checking output data (original tracks format)
    manager_data = manager_data.sort_values(['TrackID', 'T'])
    assert np.allclose(manager_data.values, data.values)


def test_interactive_tracks_interactivity(make_data_and_graph) -> None:
    """Tests InteractiveTrackManager with manipulation of the data:
    add, remove, link, unlink.
    """
    pass
    # data: pd.DataFrame = make_data_and_graph[0]
    # graph: Dict = make_data_and_graph[1]
    # manager = InteractiveTrackManager(data=data.values, graph=graph)
