import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ....utils.translations import trans
from ...utils.layer_utils import _validate_features
from ._base_track_manager import BaseTrackManager


@dataclass
class Node:
    """Node class corresponding to each indivisual row of tracks' data.
       The indexing (`data_index`) is used to query the data information from the input data.

    Attributes
    ----------
    index : int
        node's unique identifying index.
    vertex : np.array
        node's vertex coordinates T, (Z), Y, X`.
    features : pd.DataFrame
        node's features.
    parents :  List[Node]
        list of node's parents, more than one parent represents a track merge.
    children: List[Node]
        list of node's children, more than one child represents a track split (division).
    """

    index: int
    vertex: np.ndarray
    features: Optional[pd.DataFrame] = None
    parents: List['Node'] = field(default_factory=list)
    children: List['Node'] = field(default_factory=list)

    @property
    def time(self) -> int:
        return self.vertex[0]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return self.index == other.index


# FIXME: don't forget to remove this
def connex(vertices: np.ndarray) -> list:
    """Connection array to build vertex edges for vispy LineVisual.

    Notes
    -----
    See
    http://api.vispy.org/en/latest/visuals.html#vispy.visuals.LineVisual

    """
    return [True] * (vertices.shape[0] - 1) + [False]


class InteractiveTrackManager(BaseTrackManager):
    """Manage track data and simplify interactions with the Tracks layer.
    TODO: update this

    Attributes
    ----------
    data : array (N, D+1)
        Coordinates for N points in D+1 dimensions. ID,T,(Z),Y,X. The first
        axis is the integer ID of the track. D is either 3 or 4 for planar
        or volumetric timeseries respectively.
    features : Dataframe-like
        Features table where each row corresponds to a point and each column
        is a feature.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each point. Each property should be an array of length N,
        where N is the number of points.
    graph : dict {int: list}
        Graph representing associations between tracks. Dictionary defines the
        mapping between a track ID and the parents of the track. This can be
        one (the track has one parent, and the parent has >=1 child) in the
        case of track splitting, or more than one (the track has multiple
        parents, but only one child) in the case of track merging.
        See examples/tracks_3d_with_graph.py
    ndim : int
        Number of spatiotemporal dimensions of the data.
    max_time: float, int
        Maximum value of timestamps in data.
    track_vertices : array (N, D)
        Vertices for N points in D dimensions. T,(Z),Y,X
    track_connex : array (N,)
        Connection array specifying consecutive vertices that are linked to
        form the tracks. Boolean
    track_times : array (N,)
        Timestamp for each vertex in track_vertices.
    graph_vertices : array (N, D)
        Vertices for N points in D dimensions. T,(Z),Y,X
    graph_connex : array (N,)
        Connection array specifying consecutive vertices that are linked to
        form the graph.
    graph_times : array (N,)
        Timestamp for each vertex in graph_vertices.
    track_ids : array (N,)
        Track ID for each vertex in track_vertices.
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        graph: Dict = {},
        features: Optional[pd.DataFrame] = None,
    ):

        # store the raw data here
        self._ndim = 0

        # stores the nodes in each specific time point
        self._time_to_nodes: Dict[int, List[Node]] = {}

        # maps from the data row indices to their respective node
        self._id_to_nodes: Dict[int, Node] = {}
        self._max_node_index = 0

        # stores the last node of tracks, so they can be transversed backwardly
        self._tracks: Dict[int, Node] = {}

        self._features: Optional[pd.DataFrame] = None

        if data is None:
            assert len(graph) == 0 and features is None
        else:
            self._build_tracks(data, graph, features)

    def _build_tracks(
        self,
        data: Union[list, np.ndarray],
        graph: Dict,
        features: Optional[pd.DataFrame],
    ) -> None:

        columns = ['TrackID', 'T', 'Z', 'Y', 'X']

        if data.shape[1] == 4:
            columns.remove('Z')
        elif data.shape[1] != 5:
            raise RuntimeError(
                f'Tracks data must have 4 or 5 columns, found {data.shape[1]}.'
            )

        # naming input_data to avoid confusion with the actual data that needs to be computed from the nodes
        data = pd.DataFrame(data, columns=columns)
        self._ndim = data.shape[1] - 1

        self._time_to_nodes = {}
        self._tracks = {}
        self._id_to_nodes = {}

        # used make connections from `graph`
        track_to_roots: Dict[int, Node] = {}
        track_to_leafs: Dict[int, Node] = {}

        for track_id, track in data.groupby('TrackID', sort=False):
            parent_node = None
            track = track.sort_values('T')
            for index, row in track.iterrows():
                feats = None if features is None else features[index]
                node = self._add_node(index, row[1:].values, feats)

                if parent_node is not None:
                    parent_node.children.append(node)
                    node.parents.append(parent_node)
                else:
                    track_to_roots[track_id] = node

                parent_node = node

            # leaf (last) node
            self._tracks[node.index] = node
            track_to_leafs[track_id] = node

        for track_id, parents in graph.items():
            node = track_to_roots[track_id]
            for parent_track_id in parents:
                parent = track_to_leafs[parent_track_id]
                if len(parent.children) == 0:
                    # it's not a leaf anymore
                    del self._tracks[parent.index]
                node.parents.append(parent)
                parent.children.append(node)

        self._max_node_index = data.shape[0]

    @property
    def ndim(self) -> int:
        """Determine number of spatiotemporal dimensions of the layer."""
        if self._ndim == 0 and len(self._id_to_nodes) > 0:
            node = next(iter(self._id_to_nodes.values()))
            self._ndim = len(node.vertex)

        return self._ndim

    @property
    def data(self) -> np.ndarray:
        """array (N, D+1): Coordinates for N points in D+1 dimensions."""
        raise NotImplementedError

    @data.setter
    def data(self, data: Union[list, np.ndarray]):
        raise NotImplementedError

    @property
    def features(self):
        """Dataframe-like features table.

        It is an implementation detail that this is a `pandas.DataFrame`. In the future,
        we will target the currently-in-development Data API dataframe protocol [1].
        This will enable us to use alternate libraries such as xarray or cuDF for
        additional features without breaking existing usage of this.

        If you need to specifically rely on the pandas API, please coerce this to a
        `pandas.DataFrame` using `features_to_pandas_dataframe`.

        References
        ----------
        .. [1]: https://data-apis.org/dataframe-protocol/latest/API.html
        """
        raise NotImplementedError

    @features.setter
    def features(
        self,
        features: Union[Dict[str, np.ndarray], pd.DataFrame],
    ) -> None:
        features = _validate_features(features, num_data=len(self.data))
        if 'track_id' not in features:
            features['track_id'] = self.track_ids
        self._features = features.iloc[self._order].reset_index(drop=True)
        raise NotImplementedError

    @property
    def graph(self) -> Dict[int, Union[int, List[int]]]:
        """dict {int: list}: Graph representing associations between tracks."""
        raise NotImplementedError

    @graph.setter
    def graph(self, graph: Dict[int, Union[int, List[int]]]):
        """set the track graph"""
        raise NotImplementedError

    @property
    def unique_track_ids(self) -> np.ndarray:
        """return the unique track identifiers"""
        return np.array(list(self._tracks.keys()))

    def __len__(self):
        """return the number of tracks"""
        return len(self.unique_track_ids) if self.data is not None else 0

    def build_tracks(self):
        pass

    def build_graph(self):
        pass

    def vertex_properties(self, color_by: str) -> np.ndarray:
        """return the properties of tracks by vertex"""
        raise NotImplementedError

        if color_by not in self.properties:
            raise ValueError(
                trans._(
                    'Property {color_by} not found',
                    deferred=True,
                    color_by=color_by,
                )
            )

        return self.properties[color_by]

    def get_value(self, coords: np.ndarray) -> int:
        """use a kd-tree to lookup the ID of the nearest tree"""
        raise NotImplementedError

        if self._kdtree is None:
            return

        # query can return indices to points that do not exist, trim that here
        # then prune to only those in the current frame/time
        # NOTE(arl): I don't like this!!!
        d, idx = self._kdtree.query(coords, k=10)
        idx = [i for i in idx if i >= 0 and i < self._points.shape[0]]
        pruned = [i for i in idx if self._points[i, 0] == coords[0]]

        # if we have found a point, return it
        if pruned and self._points_id is not None:
            return self._points_id[pruned[0]]  # return the track ID

    @property
    def max_time(self) -> int:
        """Determine the maximum timestamp of the dataset"""
        return max(self._time_to_nodes.keys())

    @property
    def track_vertices(self) -> np.ndarray:
        """return the track vertices"""
        return self._track_vertices

    @property
    def track_connex(self) -> np.ndarray:
        """vertex connections for drawing track lines"""
        return self._track_connex

    @property
    def track_colors(self) -> np.ndarray:
        """return the vertex colors according to the currently selected
        property"""
        return self._track_colors

    @property
    def graph_vertices(self) -> np.ndarray:
        """return the graph vertices"""
        return self._graph_vertices

    @property
    def graph_connex(self):
        """vertex connections for drawing the graph"""
        return self._graph_connex

    @property
    def track_times(self) -> np.ndarray:
        """time points associated with each track vertex"""
        return self.track_vertices[:, 0]

    @property
    def graph_times(self) -> np.ndarray:
        """time points associated with each graph vertex"""
        if self.graph_vertices is not None:
            return self.graph_vertices[:, 0]
        return None

    def track_labels(self, current_time: int) -> tuple:
        """return track labels at the current time"""
        # this is the slice into the time ordered points array
        if current_time not in self._points_lookup:
            return [], []

        lookup = self._points_lookup[current_time]
        pos = self._points[lookup, ...]
        lbl = [f'ID:{i}' for i in self._points_id[lookup]]
        return lbl, pos

    def _get_node(self, index: int) -> Node:
        try:
            return self._id_to_nodes[index]
        except KeyError:
            raise KeyError(
                f'Node with index {index} not found in InteractiveTrackManager'
            )

    def link(self, child_id: int, parent_id: int) -> None:
        child = self._get_node(child_id)
        parent = self._get_node(parent_id)

        if child in parent.children:
            warnings.warn(
                f'Node {parent_id} is already a parent of {child_id}.'
            )
            return

        if len(parent.children) == 0:
            # it won't be a leaf anymore
            del self._tracks[parent_id]

        child.parents.append(parent)
        parent.children.append(child)

    def _unlink_pair(self, child: Node, parent: Node) -> None:
        child.parents.remove(parent)
        parent.children.remove(child)

        if len(parent.children) == 0:
            # it becomes a leaf it there isn't a child
            self._tracks[parent.index] == parent

    def unlink(
        self, child_id: Optional[int] = None, parent_id: Optional[int] = None
    ) -> None:
        """
        Disconnects nodes indexed by child_id and parent_id.
        If one of them is not provided it disconnects every all of its connections.
        """
        if child_id is None and parent_id is None:
            raise RuntimeError(
                '`child_id`, `parent_id` or both must be supplified.'
            )

        if child_id is None:
            parent = self._get_node(parent_id)
            for child in tuple(parent.children):
                self._unlink_pair(child, parent)

        elif parent_id is None:
            child = self._get_node(child_id)
            for parent in tuple(child.parents):
                self._unlink_pair(child, parent)

        else:
            child = self._get_node(child_id)
            parent = self._get_node(parent_id)
            self._unlink_pair(child, parent)

    def remove(self, node_index: int, keep_link: bool = False) -> None:
        node = self._get_node(node_index)

        # removing from storages
        del self._id_to_nodes[node_index]
        if len(node.children) == 0:
            del self._tracks[node_index]
        # this operation is not done in constant time
        self._time_to_nodes[node.time].remove(node)

        if keep_link:
            if len(node.children) > 1 and len(node.parents) > 1:
                # this is not allowed since the connection would be arbitrary
                raise RuntimeError(
                    'Removal with linking is not allowed for nodes with multiple'
                    + f'children ({len(node.children)}) and parents {len(node.parents)}.'
                )

            for child in node.children:
                for parent in node.parents:
                    parent.children.append(child)
                    child.parents.append(parent)

        for child in node.children:
            child.parents.remove(node)

        for parent in node.parents:
            parent.children.remove(node)
            if len(parent.children) == 0:
                # if no children exists it becomes a leaf
                self._tracks[parent.track_id] = parent

    def _add_node(
        self,
        index: int,
        vertex: np.ndarray,
        features: Optional[pd.DataFrame] = None,
    ) -> Node:
        node = Node(index=index, vertex=vertex, features=features)
        self._id_to_nodes[index] = node

        time = node.time
        if time not in self._time_to_nodes:
            self._time_to_nodes[time] = []

        self._time_to_nodes[time].append(node)

        return node

    def add(
        self,
        vertex: Union[list, np.ndarray],
        features: Optional[pd.DataFrame] = None,
    ) -> int:
        if len(vertex) != self.ndim and self.ndim != 0:
            # ndim is 0 when the data is empty
            raise RuntimeError(
                f'Vertex must match data dimension. Found {len(vertex)}, expected {self._ndim}.'
            )

        self._max_node_index += 1
        node = self._add_node(
            index=self._max_node_index, vertex=vertex, features=features
        )
        self._tracks[node.index] = node
