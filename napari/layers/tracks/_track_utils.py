from typing import Dict, List, Union

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from ...utils.translations import trans
from ..utils.layer_utils import dataframe_to_properties


def connex(vertices: np.ndarray) -> list:
    """Connection array to build vertex edges for vispy LineVisual.

    Notes
    -----
    See
    http://api.vispy.org/en/latest/visuals.html#vispy.visuals.LineVisual

    """
    return [True] * (vertices.shape[0] - 1) + [False]


class TrackManager:
    """Manage track data and simplify interactions with the Tracks layer.

    Attributes
    ----------
    data : array (N, D+1)
        Coordinates for N points in D+1 dimensions. ID,T,(Z),Y,X. The first
        axis is the integer ID of the track. D is either 3 or 4 for planar
        or volumetric timeseries respectively.
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

    Methods
    -------


    """

    def __init__(self):

        # store the raw data here
        self._data = None
        self._properties = None
        self._order = None

        # use a kdtree to help with fast lookup of the nearest track
        self._kdtree = None

        # NOTE(arl): _tracks and _connex store raw data for vispy
        self._points = None
        self._points_id = None
        self._points_lookup = None
        self._ordered_points_idx = None

        self._track_vertices = None
        self._track_connex = None
        self._track_colors = None

        self._graph = None
        self._graph_vertices = None
        self._graph_connex = None

        # lookup table for vertex indices from track id
        self._id2idxs = None

    @property
    def data(self) -> np.ndarray:
        """array (N, D+1): Coordinates for N points in D+1 dimensions."""
        return self._data

    @data.setter
    def data(self, data: Union[list, np.ndarray]):
        """set the vertex data and build the vispy arrays for display"""

        # convert data to a numpy array if it is not already one
        data = np.asarray(data)

        # Sort data by ID then time
        self._order = np.lexsort((data[:, 1], data[:, 0]))
        data = data[self._order]

        # check check the formatting of the incoming track data
        self._data = self._validate_track_data(data)

        # build the indices for sorting points by time
        self._ordered_points_idx = np.argsort(self.data[:, 1])
        self._points = self.data[self._ordered_points_idx, 1:]

        # build a tree of the track data to allow fast lookup of nearest track
        self._kdtree = cKDTree(self._points)

        # make the lookup table
        # NOTE(arl): it's important to convert the time index to an integer
        # here to make sure that we align with the napari dims index which
        # will be an integer - however, the time index does not necessarily
        # need to be an int, and the shader will render correctly.
        frames = list(set(self._points[:, 0].astype(np.uint).tolist()))
        self._points_lookup = {}
        for f in frames:
            idx = np.where(self._points[:, 0] == f)[0]
            self._points_lookup[f] = slice(min(idx), max(idx) + 1, 1)

        # make a second lookup table using a sparse matrix to convert track id
        # to the vertex indices
        self._id2idxs = coo_matrix(
            (
                np.broadcast_to(1, self.track_ids.size),  # just dummy ones
                (self.track_ids, np.arange(self.track_ids.size)),
            )
        ).tocsr()

        # sort the data by ID then time
        # indices = np.lexsort((self.data[:, 1], self.data[:, 0]))

    @property
    def properties(self) -> Dict[str, np.ndarray]:
        """dict {str: np.ndarray (N,)}, DataFrame: Properties for each track."""
        return self._properties

    @properties.setter
    def properties(self, properties: Dict[str, np.ndarray]):
        """set track properties"""

        # make copy so as not to mutate original
        properties = properties.copy()

        if not isinstance(properties, dict):
            properties, _ = dataframe_to_properties(properties)

        if 'track_id' not in properties:
            properties['track_id'] = self.track_ids

        # order properties dict
        for prop in properties.keys():
            arr = np.array(properties[prop])
            arr = arr[self._order]
            properties[prop] = arr

        # check the formatting of incoming properties data
        self._properties = self._validate_track_properties(properties)

    @property
    def graph(self) -> Dict[int, Union[int, List[int]]]:
        """dict {int: list}: Graph representing associations between tracks."""
        return self._graph

    @graph.setter
    def graph(self, graph: Dict[int, Union[int, List[int]]]):
        """set the track graph"""
        self._graph = self._validate_track_graph(graph)

    @property
    def track_ids(self):
        """return the track identifiers"""
        return self.data[:, 0].astype(np.uint32)

    @property
    def unique_track_ids(self):
        """return the unique track identifiers"""
        return np.unique(self.track_ids)

    def __len__(self):
        """return the number of tracks"""
        return len(self.unique_track_ids) if self.data is not None else 0

    def _vertex_indices_from_id(self, track_id: int):
        """return the vertices corresponding to a track id"""
        return self._id2idxs[track_id].nonzero()[1]

    def _validate_track_data(self, data: np.ndarray) -> np.ndarray:
        """validate the coordinate data"""

        if data.ndim != 2:
            raise ValueError(
                trans._('track vertices should be a NxD array', deferred=True)
            )

        if data.shape[1] < 4 or data.shape[1] > 5:
            raise ValueError(
                trans._(
                    'track vertices should be 4 or 5-dimensional',
                    deferred=True,
                )
            )

        # check that all IDs are integers
        ids = data[:, 0]
        if not np.all(np.floor(ids) == ids):
            raise ValueError(
                trans._('track id must be an integer', deferred=True)
            )

        if not all([t >= 0 for t in data[:, 1]]):
            raise ValueError(
                trans._(
                    'track timestamps must be greater than zero', deferred=True
                )
            )

        # check that data are sorted by ID then time
        indices = np.lexsort((data[:, 1], data[:, 0]))
        if not np.array_equal(indices, np.arange(data[:, 0].size)):
            raise ValueError(
                trans._(
                    'tracks should be ordered by ID and time', deferred=True
                )
            )

        return data

    def _validate_track_properties(
        self, properties: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """validate the track properties"""

        for k, v in properties.items():
            if len(v) != len(self.data):
                raise ValueError(
                    trans._(
                        'the number of properties must equal the number of vertices',
                        deferred=True,
                    )
                )
            # ensure the property values are a numpy array
            if type(v) != np.ndarray:
                properties[k] = np.asarray(v)

        return properties

    def _validate_track_graph(
        self, graph: Dict[int, Union[int, List[int]]]
    ) -> Dict[int, List[int]]:
        """validate the track graph"""

        # check that graph nodes are of correct format
        for node_idx, parents_idx in graph.items():
            # make sure parents are always a list
            if type(parents_idx) != list:
                graph[node_idx] = [parents_idx]

        # check that graph nodes exist in the track id lookup
        for node_idx, parents_idx in graph.items():
            nodes = [node_idx] + parents_idx
            for node in nodes:
                if node not in self.unique_track_ids:
                    raise ValueError(
                        trans._(
                            'graph node {node_idx} not found',
                            deferred=True,
                            node_idx=node_idx,
                        )
                    )

        return graph

    def build_tracks(self):
        """build the tracks"""

        points_id = []
        track_vertices = []
        track_connex = []

        # NOTE(arl): this takes some time when the number of tracks is large
        for idx in self.unique_track_ids:
            indices = self._vertex_indices_from_id(idx)

            # grab the correct vertices and sort by time
            vertices = self.data[indices, 1:]
            vertices = vertices[vertices[:, 0].argsort()]

            # coordinates of the text identifiers, vertices and connections
            points_id += [idx] * vertices.shape[0]
            track_vertices.append(vertices)
            track_connex.append(connex(vertices))

        self._points_id = np.array(points_id)[self._ordered_points_idx]
        self._track_vertices = np.concatenate(track_vertices, axis=0)
        self._track_connex = np.concatenate(track_connex, axis=0)

    def build_graph(self):
        """build the track graph"""

        graph_vertices = []
        graph_connex = []

        for node_idx, parents_idx in self.graph.items():
            # we join from the first observation of the node, to the last
            # observation of the parent
            node_start = self._vertex_indices_from_id(node_idx)[0]
            node = self.data[node_start, 1:]

            for parent_idx in parents_idx:
                parent_stop = self._vertex_indices_from_id(parent_idx)[-1]
                parent = self.data[parent_stop, 1:]

                verts = np.stack([node, parent], axis=0)

                graph_vertices.append(verts)
                graph_connex.append([True, False])

        # if there is a graph, store the vertices and connection arrays,
        # otherwise, clear the vertex arrays
        if graph_vertices:
            self._graph_vertices = np.concatenate(graph_vertices, axis=0)
            self._graph_connex = np.concatenate(graph_connex, axis=0)
        else:
            self._graph_vertices = None
            self._graph_connex = None

    def vertex_properties(self, color_by: str) -> np.ndarray:
        """return the properties of tracks by vertex"""

        if color_by not in self.properties:
            raise ValueError(
                trans._(
                    'Property {color_by} not found',
                    deferred=True,
                    color_by=color_by,
                )
            )

        return self.properties[color_by]

    def get_value(self, coords):
        """use a kd-tree to lookup the ID of the nearest tree"""
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
    def ndim(self) -> int:
        """Determine number of spatiotemporal dimensions of the layer."""
        return self.data.shape[1] - 1

    @property
    def max_time(self) -> int:
        """Determine the maximum timestamp of the dataset"""
        return int(np.max(self.track_times))

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
        """time points assocaite with each graph vertex"""
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
