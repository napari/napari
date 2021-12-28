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
        [  # track id, time, index (for reference)
            [1, 0],  # 0
            [1, 1],  # 1
            [1, 2],  # 2
            [1, 3],  # 3
            [2, 4],  # 4
            [2, 5],  # 5
            [2, 6],  # 6
            [2, 7],  # 7
            [2, 8],  # 8
            [2, 9],  # 9
            [3, 4],  # 10
            [3, 5],  # 11
            [3, 6],  # 12
            [4, 7],  # 13
            [4, 8],  # 14
            [4, 9],  # 15
            [5, 7],  # 16
            [5, 8],  # 17
            [5, 9],  # 18
            [6, 1],  # 19
            [6, 2],  # 20
            [6, 3],  # 21
            [7, 4],  # 22
            [7, 5],  # 23
            [8, 4],  # 24
            [8, 5],  # 25
            [9, 6],  # 26
            [9, 7],  # 27
        ]
    )

    coords = np.random.randn(len(id_and_time), 3)  # (z, y, x)
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

    assert np.all(track_connex == manager._track_connex)

    # sorting and checking output data (original tracks format)
    manager_data = manager_data.sort_values(['TrackID', 'T'])
    assert np.allclose(manager_data.values, data.values)


def test_interactive_tracks_interactivity(
    make_napari_viewer, make_data_and_graph
) -> None:
    """Tests InteractiveTrackManager with manipulation of the data:
    add, remove, link, unlink.
    """

    data: pd.DataFrame = make_data_and_graph[0]
    graph: Dict = make_data_and_graph[1]

    viewer = make_napari_viewer()
    tracks_layer = viewer.add_tracks(data=data.values, graph=graph)

    tracks_layer.interactive_mode = True
    manager = tracks_layer._manager
    assert isinstance(manager, InteractiveTrackManager)

    manager.serialize()
    assert manager._is_serialized

    manager.remove(6, keep_link=True)
    assert not manager._is_serialized
    assert len(manager._id_to_nodes) == data.shape[0] - 1
    assert len(manager._leafs) == 4
    assert 6 not in manager._id_to_nodes

    manager.remove(7, keep_link=False)
    assert len(manager._leafs) == 5
    assert len(manager._id_to_nodes) == data.shape[0] - 2
    assert 7 not in manager._id_to_nodes

    new_node_id = manager.add([4, 0.25, 0.25, 0.25])
    assert new_node_id == data.shape[0]
    assert len(manager._leafs) == 6
    assert len(manager._id_to_nodes) == data.shape[0] - 1

    manager.link(new_node_id, 17)
    new_node_parents = manager._id_to_nodes[new_node_id].parents
    assert len(manager._leafs) == 6
    assert len(manager._id_to_nodes) == data.shape[0] - 1
    assert len(new_node_parents) == 1 and new_node_parents[0].index == 17
    assert (
        manager._id_to_nodes[new_node_id] in manager._id_to_nodes[17].children
    )

    prev_children = manager._id_to_nodes[12].children
    manager.unlink(child_id=None, parent_id=12)
    assert len(manager._leafs) == 7
    assert len(manager._id_to_nodes) == data.shape[0] - 1
    assert len(manager._id_to_nodes[12].children) == 0
    assert all(12 not in child.parents for child in prev_children)

    prev_parent = manager._id_to_nodes[9].parents[0]
    manager.remove(9)
    assert 9 not in manager._leafs
    assert prev_parent.index in manager._leafs
    assert len(manager._leafs) == 7
    assert len(manager._id_to_nodes) == data.shape[0] - 2

    tracks_layer.interactive_mode = False
    assert tracks_layer.data.shape[0] == data.shape[0] - 2
