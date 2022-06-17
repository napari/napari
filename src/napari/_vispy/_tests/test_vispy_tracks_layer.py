from napari._vispy.layers.tracks import VispyTracksLayer
from napari.layers import Tracks


def test_tracks_graph_cleanup():
    """
    Test if graph data can be cleaned up without any issue.
    There was problems with the shader buffer once, see issue #4155.
    """
    tracks_data = [
        [1, 0, 236, 0],
        [1, 1, 236, 100],
        [1, 2, 236, 200],
        [2, 3, 436, 500],
        [2, 4, 436, 1000],
        [3, 3, 636, 500],
        [3, 4, 636, 1000],
    ]
    graph = {1: [], 2: [1], 3: [1]}

    layer = Tracks(tracks_data, graph=graph)
    visual = VispyTracksLayer(layer)

    layer.graph = {}

    assert visual.node._subvisuals[2]._pos is None
    assert visual.node._subvisuals[2]._connect is None
