from ..base import Layer
from ...utils.event import Event
from ...utils.colormaps.colormaps import vispy_or_mpl_colormap

from typing import Union, Tuple, List

import numpy as np

from scipy.spatial import cKDTree

# from matplotlib.cm import get_cmap


class Tracks(Layer):
    """ Tracks

    A napari-style Tracks layer for overlaying trajectories on image data.


    Parameters
    ----------

        data : list
            list of (NxD) arrays of the format: t,x,y,(z),....,n

        properties : list
            list of dictionaries of track properties:

            [{'ID': 0,
              'parent': [],
              'root': 0,
              'states': [], ...}, ...]

        colomaps : dict
            dictionary list of colormap objects to use for track
            properties:

            {'states': IndexedColormap}

    importantly, p (parent) is a list of track IDs that are parents of the
    track, this can be one (the track has one parent, and the parent has >=1
    child) in the case of splitting, or more than one (the track has multiple
    parents, but only one child) in the case of track merging

    """

    # The max number of points that will ever be used to render the thumbnail
    # If more points are present then they are randomly subsampled
    _max_tracks_thumbnail = 1024

    def __init__(
        self,
        data=None,
        *,
        properties=None,
        graph=None,
        edge_width=2,
        tail_length=30,
        color_by=0,
        n_dimensional=True,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        opacity=1,
        blending='translucent',
        visible=True,
        colormaps=None,
    ):

        # if not provided with any data, set up an empty layer in 2D+t
        if data is None:
            data = [np.empty((0, 3))]

        ndim = self._check_track_dimensionality(data)

        super().__init__(
            data,
            ndim,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            opacity=opacity,
            blending=blending,
            visible=visible,
        )

        self.events.add(
            edge_width=Event,
            edge_color=Event,
            tail_length=Event,
            display_id=Event,
            display_tail=Event,
            display_graph=Event,
            current_edge_color=Event,
            current_properties=Event,
            n_dimensional=Event,
            color_by=Event,
            properties=Event,
        )

        # store the currently displayed dims, we can use changes to this to
        # refactor what is sent to vispy
        self._current_dims_displayed = self.dims.displayed

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

        self.data = data  # this is the track data
        self.properties = properties or []
        self.colormaps = colormaps or {}

        self.edge_width = edge_width
        self.tail_length = tail_length
        self.display_id = False
        self.display_tail = True
        self.display_graph = True
        self.color_by = 'ID'  # default color by ID

        self._update_dims()

    def _get_extent(self) -> List[Tuple[int, int, int]]:
        """Determine ranges for slicing given by (min, max, step)."""

        def _minmax(x):
            return (int(np.min(x)), int(np.max(x)) + 1, 1)

        return [_minmax(self._track_vertices[:, i]) for i in range(self.ndim)]

    def _get_ndim(self) -> int:
        """Determine number of dimensions of the layer."""
        return self._track_vertices.shape[1]

    def _get_state(self):
        """Get dictionary of layer state.

        Returns
        -------
        state : dict
            Dictionary of layer state.
        """
        state = self._get_base_state()
        state.update(
            {
                'edge_width': self.edge_width,
                'tail_length': self.tail_length,
                'properties': self.properties,
                'n_dimensional': self.n_dimensional,
                'data': self.data,
            }
        )
        return state

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""

        # check whether we want to slice the data
        if self.dims.displayed != self._current_dims_displayed:
            self._current_dims_displayed = self.dims.displayed
            # TODO(arl): we can use the shader masking to slice the data

        return

    def _get_value(self):
        """ use a kd-tree to lookup the ID of the nearest tree """
        if self._kdtree is None:
            return

        # need to swap x,y for this to work
        coords = np.array(self.coordinates)
        coords[2], coords[1] = coords[1], coords[2]

        d, idx = self._kdtree.query(coords, k=10)
        pruned = [i for i in idx if self._points[i, 0] == coords[0]]
        if pruned and self._points_id is not None:
            return self._points_id[pruned[0]]  # return the track ID

    def _update_thumbnail(self):
        """Update thumbnail with current points and colors."""
        pass

    @property
    def _view_data(self):
        """ return a view of the data """
        return self._pad_display_data(self._track_vertices)

    @property
    def _view_graph(self):
        """ return a view of the graph """
        if not self._graph:
            return None
        return self._pad_display_data(self._graph_vertices)

    def _pad_display_data(self, vertices):
        """ pad display data when moving between 2d and 3d


        NOTES:
            2d data is transposed yx
            3d data is zyxt

        """
        data = vertices[:, self.dims.displayed]
        # if we're only displaying two dimensions, then pad the display dim
        # with zeros
        if self.dims.ndisplay == 2:
            data = np.pad(data, ((0, 0), (0, 1)), 'constant')
            data = data[:, (1, 0, 2)]  # y, x, z
        else:
            data = data[:, (2, 1, 0)]  # z, y, x
        return data

    @property
    def current_time(self):
        # TODO(arl): get the correct index here
        if isinstance(self.dims.indices[0], slice):
            return int(np.max(self.track_times))
        return self.dims.indices[0]

    @property
    def data(self) -> list:
        """list of (N, D) arrays: coordinates for N points in D dimensions."""
        return self._data

    @data.setter
    def data(self, data: list):
        """ set the data and build the vispy arrays for display """
        self._data = data

        # build the connex for the data
        def _cnx(d):
            return [True] * (d.shape[0] - 1) + [False]

        # build the track data for vispy
        self._track_vertices = np.concatenate(self.data, axis=0)
        self._track_connex = np.concatenate([_cnx(d) for d in data], axis=0)

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
        self._build_graph()

        # properties have been updated, we need to alert the gui
        self.events.properties()

    def _build_graph(self):
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
        for g in self._graph:
            try:
                edge = (_get_id(g[0]), _get_id(g[1]))
                graph.append(edge)
            except IndexError:
                continue

        # if we have no graph, return
        if not graph:
            return

        # we can use the graph to build the vertices and edges of the graph
        vertices = []
        connex = []

        for node_idx, parent_idx in graph:
            # we join from the first observation of the node, to the last
            # observation of the parent
            node = self.data[node_idx][0, ...]
            parent = self.data[parent_idx][-1, ...]

            verts = np.stack([node, parent], axis=0)
            vertices.append(verts)

            connex.append([True, False])

        self._graph_vertices = np.concatenate(vertices, axis=0)
        self._graph_connex = np.concatenate(connex, axis=0)

    @property
    def graph(self) -> list:
        """ return the graph """
        return self._graph

    @property
    def edge_width(self) -> Union[int, float]:
        """float: Width for all vectors in pixels."""
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: Union[int, float]):
        self._edge_width = edge_width
        self.events.edge_width()
        self.refresh()
        # self.status = format_float(self.edge_width)

    @property
    def tail_length(self) -> Union[int, float]:
        """float: Width for all vectors in pixels."""
        return self._tail_length

    @tail_length.setter
    def tail_length(self, tail_length: Union[int, float]):
        self._tail_length = tail_length
        self.events.tail_length()
        self.refresh()
        # self.status = format_float(self.edge_width)

    @property
    def display_id(self) -> bool:
        return self._display_id

    @display_id.setter
    def display_id(self, value: bool):
        self._display_id = value
        self.events.display_id()
        self.refresh()

    @property
    def display_tail(self) -> bool:
        return self._display_tail

    @display_tail.setter
    def display_tail(self, value: bool):
        self._display_tail = value
        self.events.display_tail()
        self.refresh()

    @property
    def display_graph(self) -> bool:
        return self._display_graph

    @display_graph.setter
    def display_graph(self, value: bool):
        self._display_graph = value
        self.events.display_tail()
        self.refresh()

    @property
    def color_by(self) -> str:
        return self._color_by

    @color_by.setter
    def color_by(self, color_by: str):
        self._color_by = color_by

        # if we change the coloring, rebuild the vertex colors array
        vertex_properties = []
        for idx, track_property in enumerate(self.properties):
            property = track_property[self.color_by]

            if isinstance(property, (list, np.ndarray)):
                p = property
            elif isinstance(property, (int, float, np.generic)):
                p = [property] * len(self.data[idx])  # length of the track
            else:
                raise TypeError(
                    'Property {track_property} type not recognized'
                )

            vertex_properties.append(p)

        # concatenate them, and use a colormap to color them
        vertex_properties = np.concatenate(vertex_properties, axis=0)

        def _norm(p):
            return (p - np.min(p)) / np.max([1e-10, np.ptp(p)])

        # TODO(arl): remove matplotlib dependency?
        if self.color_by in self.colormaps:
            colormap = self.colormaps[self.color_by]
        else:
            # if we don't have a colormap, get one and scale the properties
            colormap = vispy_or_mpl_colormap('hsv')
            vertex_properties = _norm(vertex_properties)

        # actually set the vertex colors
        self._track_colors = colormap[vertex_properties]

        # fire the events and update the display
        self.events.color_by()
        self.refresh()

    def _check_track_dimensionality(self, data: list):
        """ check the dimensionality of the data

        TODO(arl): we could allow a mix of 2D/3D etc...
        """
        assert all([isinstance(d, np.ndarray) for d in data])
        assert all([d.shape[1] == data[0].shape[1] for d in data])
        return data[0].shape[1]

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
    def graph_connex(self):
        """ vertex connections for drawing the graph """
        return self._graph_connex

    @property
    def track_times(self) -> np.ndarray:
        """ time points associated with each track vertex """
        return self._track_vertices[:, 0]

    @property
    def graph_times(self) -> np.ndarray:
        """ time points assocaite with each graph vertex """
        if self._graph:
            return self._graph_vertices[:, 0]
        return None

    @property
    def track_labels(self):
        """ return track labels at the current time """
        # this is the slice into the time ordered points array
        lookup = self._points_lookup[self.current_time]
        # TODO(arl): this breaks when changing dimensions
        pos = self._pad_display_data(self._points[lookup, ...])
        lbl = [f'ID:{i}' for i in self._points_id[lookup]]
        return zip(lbl, pos)
