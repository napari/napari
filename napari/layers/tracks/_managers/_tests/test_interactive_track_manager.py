from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from napari.layers.tracks._managers import InteractiveTrackManager
from napari.layers.tracks._managers._track_manager import connex


@pytest.fixture
def tracks_data_and_graph() -> Tuple[pd.DataFrame, Dict]:
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


def track_id_mapping(data: pd.DataFrame) -> Dict[int, int]:
    mapping = {}
    for track_id, track in data.groupby('TrackID'):
        track = track.sort_values('T', ascending=False)
        node_index, _ = next(track.iterrows())
        mapping[node_index] = int(track_id)
    return mapping


def test_interactive_tracks_init(tracks_data_and_graph) -> None:
    """Tests InteractiveTrackManager data structure construction and serialization."""
    data: pd.DataFrame = tracks_data_and_graph[0]
    graph: Dict = tracks_data_and_graph[1]
    manager = InteractiveTrackManager(data=data.values, graph=graph)

    # checking if manager is storing the data as expected
    assert manager.ndim == 4  # (t, z, y, x)
    assert manager.max_time == data['T'].max()
    assert len(manager._id_to_nodes) == data.shape[0]
    assert len(manager._leafs) == 4  # number of leafs
    assert not manager._is_serialized

    mapping = track_id_mapping(data)

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
    make_napari_viewer, tracks_data_and_graph
) -> None:
    """Tests InteractiveTrackManager with manipulation of the data:
    add, remove, link, unlink.
    """

    data: pd.DataFrame = tracks_data_and_graph[0]
    graph: Dict = tracks_data_and_graph[1]

    viewer = make_napari_viewer()
    tracks_layer = viewer.add_tracks(data=data.values, graph=graph)

    tracks_layer.interactive_mode = True
    manager = tracks_layer._manager
    assert isinstance(manager, InteractiveTrackManager)

    manager.serialize()
    assert manager._is_serialized

    r"""
    time   | 0 1 2 3 4 5 6 7 8 9
    ----------------------------
    tracks | 1-1-1-1-2-2---2-2-2
           |       \-3-3-3-4-4-4
           |             \-5-5-5
           |   6-6-6-7-7-9-9
           |       \-8-8-/
    """
    manager.remove(6, keep_link=True)
    assert not manager._is_serialized
    assert len(manager._id_to_nodes) == data.shape[0] - 1
    assert len(manager._leafs) == 4
    assert 6 not in manager._id_to_nodes

    r"""
    time   | 0 1 2 3 4 5 6 7 8 9
    ----------------------------
    tracks | 1-1-1-1-2-2
           |       \-3-3-3-4-4-4
           |             \-5-5-5
           |   6-6-6-7-7-9-9
           |       \-8-8-/
           |                10-10
    """
    manager.remove(7, keep_link=False)
    assert len(manager._leafs) == 5
    assert len(manager._id_to_nodes) == data.shape[0] - 2
    assert 7 not in manager._id_to_nodes

    r"""
    time   | 0 1 2 3 4 5 6 7 8 9
    ----------------------------
    tracks | 1-1-1-1-2-2
           |       \-3-3-3-4-4-4
           |             \-5-5-5
           |   6-6-6-7-7-9-9
           |       \-8-8-/
           |                10-10
           |                11
    """
    new_node_id = manager.add([8, 0.25, 0.25, 0.25])
    assert new_node_id == data.shape[0]
    assert len(manager._leafs) == 6
    assert len(manager._id_to_nodes) == data.shape[0] - 1

    r"""
    time   | 0 1 2 3 4 5 6 7 8 9
    ----------------------------
    tracks | 1-1-1-1-2-2
           |       \-3-3-3-4-4-4
           |             \12-5-5   <-- tracklet are split in divisions
           |               \11
           |   6-6-6-7-7-9-9
           |       \-8-8-/
           |                10-10
    """
    manager.link(new_node_id, 16)
    new_node_parents = manager._id_to_nodes[new_node_id].parents
    assert len(manager._leafs) == 6
    assert len(manager._id_to_nodes) == data.shape[0] - 1
    assert len(new_node_parents) == 1 and new_node_parents[0].index == 16
    assert (
        manager._id_to_nodes[new_node_id] in manager._id_to_nodes[16].children
    )

    r"""
    time   | 0 1 2 3 4 5 6 7 8 9
    ----------------------------
    tracks | 1-1-1-1-2-2
           |       \-3-3-3
           |               4-4-4
           |              12-5-5
           |               \11
           |   6-6-6-7-7-9-9
           |       \-8-8-/
           |                10-10
    """
    prev_children = manager._id_to_nodes[12].children
    manager.unlink(child_id=None, parent_id=12)
    assert len(manager._leafs) == 7
    assert len(manager._id_to_nodes) == data.shape[0] - 1
    assert len(manager._id_to_nodes[12].children) == 0
    assert all(12 not in child.parents for child in prev_children)

    r"""
    time   | 0 1 2 3 4 5 6 7 8 9
    ----------------------------
    tracks | 1-1-1-1-2-2
           |       \-3-3-3
           |               4-4-4
           |              12-5-5
           |               \11
           |   6-6-6-7-7-9-9
           |       \-8-8-/
           |                10
    """
    # leafs should be update with leaf removal
    prev_parent = manager._id_to_nodes[9].parents[0]
    manager.remove(9)
    assert 9 not in manager._leafs
    assert prev_parent.index in manager._leafs
    assert len(manager._leafs) == 7
    assert len(manager._id_to_nodes) == data.shape[0] - 2

    r"""
    time   | 0 1 2 3  4  5 6 7 8 9
    ------------------------------
    tracks | 1-1-1-1--2--2
           |       \-13-13-3
           |              -\
           |   6-6-6--7--7-9-9
           |       \--8--8-/
           |                 4-4-4
           |                12-5-5
           |                 \11
           |                  10
    """
    # testing a triple merge
    manager.link(26, 11)  # track id 3 merging (and spliting) to 9
    assert len(manager._leafs) == 7
    assert len(manager._id_to_nodes) == data.shape[0] - 2
    assert len(manager._id_to_nodes[26].parents) == 3
    assert len(manager._id_to_nodes[11].children) == 2
    assert manager._id_to_nodes[11] in manager._id_to_nodes[26].parents

    # testing the track layer can return to default TrackManager
    assert not manager._is_serialized
    tracks_layer.interactive_mode = False
    assert manager._is_serialized
    assert tracks_layer.data.shape[0] == data.shape[0] - 2

    # testing relabel behavior with tracks not present in the original data
    mapping = track_id_mapping(data)
    manager.relabel_track_ids(mapping)

    # track ids not present in the in the original data receives
    # an **arbitrary id** when relabeling. Hence, it could change
    # if the InteractiveTrackManager serialization ordering is updated
    r"""
    time   | 0 1 2 3  4  5 6 7 8 9
    ------------------------------
    tracks | 1-1-1-1-13-13
           |       \-10-10-3
           |              -\
           |   6-6-6--7--7-9-9
           |       \--8--8-/
           |                 4-4-4
           |                11-5-5
           |                 \12
           |                  14
    """
    # checking final results
    expected_count = {
        1: 4,
        3: 1,
        4: 3,
        5: 2,
        6: 3,
        7: 2,
        8: 2,
        9: 2,
        10: 2,
        11: 1,
        12: 1,
        13: 2,
        14: 1,
    }

    track_ids, counts = np.unique(manager.track_ids, return_counts=True)
    assert len(track_ids) == 13
    for i, count in zip(track_ids, counts):
        assert expected_count[i] == count, f'track id {i}'

    expected_graph = {
        13: [1],
        10: [1],
        3: [10],
        9: [7, 8, 10],
        7: [6],
        8: [6],
        5: [11],
        12: [11],
    }
    assert manager.graph.keys() == expected_graph.keys()
    for k, v in manager.graph.items():
        assert sorted(v) == sorted(expected_graph[k]), f'track id {k}'


def test_interactive_tracks_errors(tracks_data_and_graph) -> None:
    """Tests InteractiveTrackManager error handling"""
    data: pd.DataFrame = tracks_data_and_graph[0]
    graph: Dict = tracks_data_and_graph[1]
    manager = InteractiveTrackManager(data=data.values, graph=graph)

    # moving a node that messes with the graph ordering
    with pytest.raises(ValueError):
        # time from 0 to 5
        manager.update(node_index=0, vertex=[5, 1, 1, 1])

    # update with the wrong vertex dimension
    with pytest.raises(ValueError):
        manager.update(node_index=0, vertex=[5, 1, 1])

    # addition with the wrong vertex dimension
    with pytest.raises(ValueError):
        manager.add(vertex=[5, 1, 1, 1, 1])

    # link that messes with the graph ordering
    with pytest.raises(ValueError):
        # t: 8 <- 3
        manager.link(21, 8)

    # unlink nodes that are not connected
    with pytest.raises(ValueError):
        manager.unlink(4, 24)

    # unlink with inverted child-parent relationship
    with pytest.raises(ValueError):
        manager.unlink(3, 10)
