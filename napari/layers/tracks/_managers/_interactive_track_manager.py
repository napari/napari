import functools
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

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

    def __repr__(self) -> str:
        return f'<Node> id: {self.index} vertex: {self.vertex}'


def outdate_serialization(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self._is_serialized = False
        return method(self, *args, **kwargs)

    return wrapper


def update_serialization(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self._is_serialized:
            self.serialize()
        return method(self, *args, **kwargs)

    return wrapper


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
        self._leafs: Dict[int, Node] = {}

        self._features: Optional[pd.DataFrame] = None

        if data is None:
            assert len(graph) == 0 and features is None
        else:
            self.set_data(data, graph, features)

        # stores if serialization of tracks is up to date
        self._is_serialized = False

    @outdate_serialization
    def set_data(
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
        self._leafs = {}
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
            self._leafs[node.index] = node
            track_to_leafs[track_id] = node

        for track_id, parents in graph.items():
            node = track_to_roots[track_id]
            for parent_track_id in parents:
                parent = track_to_leafs[parent_track_id]
                if len(parent.children) == 0:
                    # it's not a leaf anymore
                    del self._leafs[parent.index]
                node.parents.append(parent)
                parent.children.append(node)

        self._max_node_index = max(self._id_to_nodes.keys())

    @property
    def ndim(self) -> int:
        """Determine number of spatiotemporal dimensions of the layer."""
        if self._ndim == 0 and len(self._id_to_nodes) > 0:
            node = next(iter(self._id_to_nodes.values()))
            self._ndim = len(node.vertex)

        return self._ndim

    @property
    @update_serialization
    def data(self) -> np.ndarray:
        """array (N, D+1): Coordinates for N points in D+1 dimensions."""
        data = np.concatenate(
            (self._track_ids[:, None], self._vertices), axis=1
        )
        return data

    @data.setter
    def data(self, data: Union[list, np.ndarray]) -> None:
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
        raise NotImplementedError
        features = _validate_features(features, num_data=len(self.data))
        if 'track_id' not in features:
            features['track_id'] = self.track_ids
        self._features = features.iloc[self._order].reset_index(drop=True)

    @property
    @update_serialization
    def graph(self) -> Dict[int, Union[int, List[int]]]:
        """dict {int: list}: Graph representing associations between tracks."""
        return self._graph

    @graph.setter
    def graph(self, graph: Dict[int, Union[int, List[int]]]):
        """set the track graph"""
        raise NotImplementedError

    @property
    def unique_track_ids(self) -> np.ndarray:
        """return the unique track identifiers"""
        return np.array(list(self._leafs.keys()))

    def __len__(self):
        """return the number of tracks"""
        return len(self.unique_track_ids)

    def build_tracks(self) -> None:
        pass

    def build_graph(self) -> None:
        pass

    def serialize(self) -> None:
        self._graph: Dict[int, List[int]] = {}

        track_ids = []
        vertices = []
        connex = []
        features = []

        queue = list(self._leafs.values())
        seen = set()  # seen track ids
        while queue:
            node = queue.pop()
            track_id = node.index
            seen.add(track_id)

            while True:
                if len(node.children) > 1:
                    # if it has multiple children it's split into multiple tracks
                    # as it's done with the default TrackManager
                    if track_id not in self._graph:
                        self._graph[track_id] = [node.index]
                    else:
                        self._graph[track_id].append(node.index)

                    connex[-1] = False
                    if node.index in seen:
                        # parent have already been computed
                        break
                    else:
                        # start a new tracklet
                        track_id = node.index
                        seen.add(track_id)

                track_ids.append(track_id)
                vertices.append(node.vertex)
                if node.features:
                    features.append(node.features)

                if len(node.parents) == 0:
                    # if orphan stop the backtracking
                    connex.append(False)
                    break
                elif len(node.parents) == 1:
                    # if it has single parent continue as usual
                    node = node.parents[0]
                    connex.append(True)
                else:
                    # if it has multiple parents it starts a new connection and append indices to graph
                    if node.index not in self._graph:
                        self._graph[track_id] = []
                    for parent in node.parents:
                        if parent.index not in seen:
                            queue.append(parent)
                        self._graph[track_id].append(parent.index)
                    connex.append(False)
                    break

        self._track_ids = np.array(track_ids)
        self._vertices = np.array(vertices)
        self._connex = np.array(connex)
        self._features = None if len(features) == 0 else pd.DataFrame(features)
        self._is_serialized = True

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

    def get_value(self, coords: np.ndarray) -> Optional[int]:
        """lookup the index of the nearest node"""
        # FIXME: add in the PR that this is not the default behavior, it returned the track id
        assert len(coords) == self.ndim

        time = int(round(coords[0]))
        nodes = self._time_to_nodes.get(time, [])
        if len(nodes) == 0:
            return None

        vertices = np.stack([node.vertex for node in nodes])
        diff = np.linalg.norm(vertices[:, 1:] - coords[None, 1:], axis=1)
        index = np.argmin(diff)

        return nodes[index].index

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

    def track_labels(self, current_time: int) -> Tuple[List, np.ndarray]:
        """return node labels at the current time"""
        # FIXME: add in the PR that this is not the default behavior, it returned the track id
        if current_time not in self._time_to_nodes:
            return [], []

        coordinates = np.stack(
            [node.vertex for node in self._time_to_nodes[current_time]]
        )
        labels = [
            f'ID:{node.index}' for node in self._id_to_nodes[current_time]
        ]
        return labels, coordinates

    def _get_node(self, index: int) -> Node:
        try:
            return self._id_to_nodes[index]
        except KeyError:
            raise KeyError(
                f'Node with index {index} not found in InteractiveTrackManager'
            )

    @outdate_serialization
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
            del self._leafs[parent_id]

        child.parents.append(parent)
        parent.children.append(child)

    def _unlink_pair(self, child: Node, parent: Node) -> None:
        child.parents.remove(parent)
        parent.children.remove(child)

        if len(parent.children) == 0:
            # it becomes a leaf it there isn't a child
            self._leafs[parent.index] == parent

    @outdate_serialization
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

    @outdate_serialization
    def remove(self, node_index: int, keep_link: bool = False) -> None:
        node = self._get_node(node_index)

        # removing from storages
        del self._id_to_nodes[node_index]
        if len(node.children) == 0:
            del self._leafs[node_index]
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
                self._leafs[parent.track_id] = parent

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

    @outdate_serialization
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
        self._leafs[node.index] = node

    @update_serialization
    def relabel_track_ids(self, mapping: Dict[int, int]) -> None:
        max_out = max(mapping.values()) + 1

        def get_value(index: int) -> int:
            if index not in mapping:
                nonlocal max_out
                mapping[index] = max_out
                max_out += 1
            return mapping[index]

        new_graph = {}
        for node, parents in self._graph.items():
            new_graph[get_value(node)] = [get_value(p) for p in parents]
        self._graph = new_graph

        vmap = np.vectorize(get_value)
        self._track_ids = vmap(self._track_ids)
