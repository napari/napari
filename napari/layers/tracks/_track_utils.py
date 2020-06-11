import numpy as np
from scipy.spatial import cKDTree


def connex(vertices: np.ndarray) -> list:
    """ make vertex edges for vispy Line """
    return [True] * (vertices.shape[0] - 1) + [False]


def check_track_dimensionality(data: list):
    """ check the dimensionality of the data

    TODO(arl): we could allow a mix of 2D/3D etc...
    """
    assert all([isinstance(d, np.ndarray) for d in data])
    assert all([d.shape[1] == data[0].shape[1] for d in data])
    return data[0].shape[1]


class TrackManager:
    """ TrackManager

    Class to manage the track data and simplify interactions with the Tracks
    layer.


    Properties:
        data
        properties
        points

        track_vertices
        track_connex
        track_times
        track_labels

        graph_vertices
        graph_connex
        graph_times

    """

    def __init__(self):

        # store the raw data here
        self._data = None
        self._properties = None

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
        self._graph_colors = None

    @property
    def data(self) -> list:
        """list of (N, D) arrays: coordinates for N points in D dimensions."""
        return self._data

    @data.setter
    def data(self, data: list):
        """ set the data and build the vispy arrays for display """
        self._data = data

        # build the track data for vispy
        self._track_vertices = np.concatenate(self.data, axis=0)
        self._track_connex = np.concatenate([connex(d) for d in data], axis=0)

        # build the indices for sorting points by time
        self._ordered_points_idx = np.argsort(self._track_vertices[:, 0])
        self._points = self._track_vertices[self._ordered_points_idx]

        # build a tree of the track data to allow fast lookup of nearest track
        self._kdtree = cKDTree(self._points)

        # make the lookup
        frames = list(set(self._points[:, 0].astype(np.uint).tolist()))
        self._points_lookup = [None] * (max(frames) + 1)
        for f in range(max(frames) + 1):
            # if we have some data for this frame, calculate the slice required
            if f in frames:
                idx = np.where(self._points[:, 0] == f)[0]
                self._points_lookup[f] = slice(min(idx), max(idx) + 1, 1)

    @property
    def properties(self) -> list:
        """ return the list of track properties """
        return self._properties

    @properties.setter
    def properties(self, properties: list):
        """ set track properties """
        assert not properties or len(properties) == len(self.data)

        if not properties:
            properties = [{'ID': i} for i in range(len(self.data))]

        points_id = []

        # do some type checking/enforcing
        for idx, track in enumerate(properties):
            for key, value in track.items():

                if isinstance(value, np.generic):
                    track[key] = value.tolist()

            # if there is not a track ID listed, generate one on the fly
            if 'ID' not in track:
                properties[idx]['ID'] = idx

            points_id += [track['ID']] * len(self.data[idx])  # track length

        self._properties = properties
        self._points_id = np.array(points_id)[self._ordered_points_idx]

        # TODO(arl): not all tracks are guaranteed to have the same keys
        self._property_keys = list(properties[0].keys())

        # build the track graph
        self.build_graph()

        # # properties have been updated, we need to alert the gui
        # self.events.properties()

    def build_graph(self):
        """ build_graph

        Build the track graph using track properties. The track graph should be:

            [(track_idx, (parent_idx,...)),...]

        """

        # if we don't have any properties, then return gracefully
        if not self.properties:
            return

        if 'parent' not in self._property_keys:
            return

        track_lookup = [track['ID'] for track in self.properties]
        track_parents = [track['parent'] for track in self.properties]

        # now remove any root nodes
        branches = zip(track_lookup, track_parents)
        self._graph = [b for b in branches if b[0] != b[1]]

        # TODO(arl): parent can also be a list in the case of merging
        # need to deal with that here

        # lookup the actual indices for the tracks
        def _get_id(x):
            return track_lookup.index(x)

        graph = []
        for node, parent in self._graph:
            try:
                edge = (_get_id(node), _get_id(parent))
                graph.append(edge)
            except ValueError:
                continue

        # if we have no graph, return
        if not graph:
            return

        # we can use the graph to build the vertices and edges of the graph
        graph_vertices = []
        graph_connex = []

        for node_idx, parent_idx in graph:
            # we join from the first observation of the node, to the last
            # observation of the parent
            node = self.data[node_idx][0, ...]
            parent = self.data[parent_idx][-1, ...]

            verts = np.stack([node, parent], axis=0)
            graph_vertices.append(verts)

            graph_connex.append([True, False])

        self._graph_vertices = np.concatenate(graph_vertices, axis=0)
        self._graph_connex = np.concatenate(graph_connex, axis=0)

    def vertex_properties(self, color_by: str) -> np.ndarray:
        """ return the properties of tracks by vertex """

        # if we change the coloring, rebuild the vertex colors array
        vertex_properties = []
        for idx, track_property in enumerate(self.properties):
            property = track_property[color_by]

            if isinstance(property, (list, np.ndarray)):
                p = property
            elif isinstance(property, (int, float, np.generic)):
                p = [property] * len(self.data[idx])  # length of the track
            else:
                raise TypeError(
                    f'Property {track_property} type not recognized'
                )

            vertex_properties.append(p)

        # concatenate them, and use a colormap to color them
        vertex_properties = np.concatenate(vertex_properties, axis=0)

        return vertex_properties

    def get_value(self, coords):
        """ use a kd-tree to lookup the ID of the nearest tree """
        if self._kdtree is None:
            return

        # need to swap x,y for this to work
        coords[2], coords[1] = coords[1], coords[2]

        d, idx = self._kdtree.query(coords, k=10)
        pruned = [i for i in idx if self._points[i, 0] == coords[0]]
        if pruned and self._points_id is not None:
            return self._points_id[pruned[0]]  # return the track ID

    @property
    def extent(self):
        """Determine ranges for slicing given by (min, max, step)."""

        def _minmax(x):
            return (int(np.min(x)), int(np.max(x)) + 1, 1)

        return [_minmax(self._track_vertices[:, i]) for i in range(self.ndim)]

    @property
    def ndim(self):
        """Determine number of dimensions of the layer."""
        return self._track_vertices.shape[1]

    @property
    def max_time(self):
        return int(np.max(self.track_times))

    @property
    def track_vertices(self) -> np.ndarray:
        return self._track_vertices

    @property
    def track_connex(self) -> np.ndarray:
        """ vertex connections for drawing track lines """
        return self._track_connex

    @property
    def track_colors(self) -> np.ndarray:
        """ return the vertex colors according to the currently selected
        property """
        return self._track_colors

    @property
    def graph_vertices(self) -> np.ndarray:
        return self._graph_vertices

    @property
    def graph_connex(self):
        """ vertex connections for drawing the graph """
        return self._graph_connex

    @property
    def track_times(self) -> np.ndarray:
        """ time points associated with each track vertex """
        return self._track_vertices[:, 0]

    @property
    def graph(self) -> list:
        """ return the graph """
        return self._graph

    @property
    def graph_times(self) -> np.ndarray:
        """ time points assocaite with each graph vertex """
        if self._graph:
            return self._graph_vertices[:, 0]
        return None

    def track_labels(self, current_time: int) -> tuple:
        """ return track labels at the current time """
        # this is the slice into the time ordered points array
        lookup = self._points_lookup[current_time]
        pos = self._points[lookup, ...]
        lbl = [f'ID:{i}' for i in self._points_id[lookup]]
        return lbl, pos

    def _build_graph(self):
        pass
