import numpy as np
from scipy.spatial import cKDTree


def connex(vertices: np.ndarray) -> list:
    """ make vertex edges for vispy Line """
    return [True] * (vertices.shape[0] - 1) + [False]


def get_track_dimensionality(data: list):
    """ check the dimensionality of the data

    TODO(arl): we could allow a mix of 2D/3D etc...
    TODO(arl): raise appropriate errors/warnings
    """

    # check that the data dimensionality is the same for all tracks
    assert all([d.shape[1] == data[0].shape[1] for d in data])

    # return the number of tracks
    return data[0].shape[1]


def validate_track_data(data: list):
    """ check the dimensionality of the data

    TODO(arl): we could allow a mix of 2D/3D etc...
    TODO(arl): raise appropriate errors/warnings

    TODO(arl): move this to the TrackManager class?
    """

    # check that the data are provided as numpy arrays
    assert all([isinstance(d, np.ndarray) for d in data])

    # check that the data dimensionality is the same for all tracks
    assert all([d.shape[1] == data[0].shape[1] for d in data])

    # check that tracks all have monotonically increasing timestamps
    assert all([all(d[:, 0] == np.maximum.accumulate(d[:, 0])) for d in data])

    # check that we don't have duplicate timestamps in any track
    assert all([d[:, 0].shape[0] == np.unique(d[:, 0]).shape[0] for d in data])


def validate_track_properties(properties: list):
    """ check the properties of the data

    TODO(arl): move this to the TrackManager class?
    """

    if not properties:
        return

    # make sure that each track has a dictionary for properties
    assert all([isinstance(p, dict) for p in properties])

    # ensure that we have the same keys for each dictionary
    property_keys = properties[0].keys()
    assert all([p.keys() == property_keys for p in properties])

    # ensure that each property is a string
    assert all([isinstance(k, str) for k in property_keys])


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

        # check check the formatting of the incoming track data
        validate_track_data(data)

        self._data = data

        # build the track data for vispy
        self._track_vertices = np.concatenate(self.data, axis=0)
        self._track_connex = np.concatenate([connex(d) for d in data], axis=0)

        # build the indices for sorting points by time
        self._ordered_points_idx = np.argsort(self._track_vertices[:, 0])
        self._points = self._track_vertices[self._ordered_points_idx]

        # build a tree of the track data to allow fast lookup of nearest track
        self._kdtree = cKDTree(self._points)

        # make the lookup table
        # NOTE(arl): it's important to convert the time index to an integer
        # here to make sure that we align with the napari dims index which
        # will be an integer - however, the time index does not necessarily
        # need to be an int, and the shader will render correctly.
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

        # check the formatting of incoming properties data
        validate_track_properties(properties)

        # make sure that we either have no data or that there is enough data
        # given the track data
        assert not properties or len(properties) == len(self.data)

        # if there are no properties, add the track ID as a minimum
        if not properties:
            properties = [{'ID': i} for i in range(len(self.data))]

        points_id = []

        # do some type checking/enforcing
        for idx, track in enumerate(properties):

            # length of this track
            track_len = len(self.data[idx])

            # if there is not a track ID listed, generate one on the fly
            if 'ID' not in track:
                properties[idx]['ID'] = idx

            # check whether the property is a scalar or list/array,
            # if list/array, ensure that the length of the list is equal to the
            # length of the track
            for key, value in track.items():
                if isinstance(value, (np.ndarray, np.generic)):
                    track[key] = value.tolist()

                if isinstance(track[key], list):
                    property_len = len(track[key])
                    if property_len != track_len:
                        raise ValueError(
                            f'Track property {key} has incorrect '
                            f'length: {property_len} (vs {track_len})'
                        )

            points_id += [track['ID']] * track_len  # track length

        # set the properties
        self._properties = properties

        # these are the positions for plotting text labels for the track IDs at
        # the correct positions in time and space
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

        # query can return indices to points that do not exist, trim that here
        # then prune to only those in the current frame/time
        d, idx = self._kdtree.query(coords, k=10)
        idx = [i for i in idx if i >= 0 and i < self._points.shape[0]]
        pruned = [i for i in idx if self._points[i, 0] == coords[0]]

        # if we have found a point, return it
        if pruned and self._points_id is not None:
            return self._points_id[pruned[0]]  # return the track ID

    @property
    def extent(self):
        """Determine ranges for slicing given by (min, max, step)."""

        def _minmax(x):
            return (np.floor(np.min(x)), np.ceil(np.max(x)))

        extrema = np.zeros((2, self.ndim))
        for dim in range(self.ndim):
            extrema[:, dim] = _minmax(self._track_vertices[:, dim])
        return extrema

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
