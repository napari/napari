# from napari.layers.base.base import Layer
# from napari.utils.event import Event
# from napari.utils.colormaps import AVAILABLE_COLORMAPS

from ..base import Layer
from ...utils.event import Event
from ...utils.colormaps import AVAILABLE_COLORMAPS

from typing import Union, Tuple, List

import numpy as np

from ._track_utils import TrackManager, check_track_dimensionality


class Tracks(Layer):
    """ Tracks

    A napari-style Tracks layer for overlaying trajectories on image data.


    Parameters
    ----------

        data : list
            list of (NxD) arrays of the format: t,x,y,(z),....,n

        properties : list (optional)

            list of dictionaries of track properties:

            [{'ID': 0,
              'parent': [],
              'root': 0,
              'states': [], ...}, ...]

            List needs to be the same length as data, and all items need to
            contain the same dictionary keys etc...

            If no properties are provided, autogenerates the ID property
            based on the track index in the list. Properties can have any
            numeric type (scalar, array), although not much type checking
            at the moment.

            importantly, parent is a list of track IDs that are parents of the
            track, this can be one (the track has one parent, and the parent
            has >=1 child) in the case of splitting, or more than one (the
            track has multiple parents, but only one child) in the case of
            track merging.

        color_by: str
            track property to color vertices by

        colormap: str:
            default colormap to use for vertex coloring

        colomaps_dict : dict
            dictionary list of colormap objects to use for coloring by track
            properties:

            colormaps are any object with a __getitem__, that return RGBA

            {'states': IndexedColormap}


    Notes
    -----

        The TrackManager class abstracts the input data types to this layer,
        meaning that it should be possible to change the data types and maintain
        the general structure of the layer.

    """

    # The max number of points that will ever be used to render the thumbnail
    # If more points are present then they are randomly subsampled
    _max_tracks_thumbnail = 1024

    def __init__(
        self,
        data=None,
        *,
        properties=None,
        edge_width=2,
        tail_length=30,
        n_dimensional=True,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        opacity=1,
        blending='translucent',
        visible=True,
        colormap='viridis',
        color_by='ID',
        colormaps_dict=None,
    ):

        # if not provided with any data, set up an empty layer in 2D+t
        if data is None:
            data = [np.empty((0, 3))]

        ndim = check_track_dimensionality(data)

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
            colormap=Event,
            properties=Event,
        )

        # store the currently displayed dims, we can use changes to this to
        # refactor what is sent to vispy
        self._current_dims_displayed = self.dims.displayed

        # track manager deals with data slicing, graph building an properties
        self._manager = TrackManager()
        self._track_colors = (
            None  # this layer takes care of coloring the tracks
        )

        self.data = data  # this is the track data
        self.properties = properties or []
        self.colormaps_dict = colormaps_dict or {}  # additional colormaps

        self.edge_width = edge_width
        self.tail_length = tail_length
        self.display_id = False
        self.display_tail = True
        self.display_graph = True
        self._color_by = color_by  # default color by ID
        self.colormap = colormap

        self._update_dims()

    def _get_extent(self) -> List[Tuple[int, int, int]]:
        """Determine ranges for slicing given by (min, max, step)."""
        return self._manager.extent

    def _get_ndim(self) -> int:
        """Determine number of dimensions of the layer."""
        return self._manager.ndim

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
                'display_id': self.display_id,
                'display_tail': self.display_tail,
                'display_graph': self.display_graph,
                'color_by': self.color_by,
                'colormap': self.colormap,
            }
        )
        return state

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""

        # check whether we want to slice the data
        if self.dims.displayed != self._current_dims_displayed:
            self._current_dims_displayed = self.dims.displayed
            # TODO(arl): we can use the shader masking to slice the data
            print(self.dims.displayed, self.dims.indices)

        return

    def _get_value(self):
        """ use a kd-tree to lookup the ID of the nearest tree """
        coords = np.array(self.coordinates)
        return self._manager.get_value(coords)

    def _update_thumbnail(self):
        """Update thumbnail with current points and colors."""
        pass

    @property
    def _view_data(self):
        """ return a view of the data """
        return self._pad_display_data(self._manager.track_vertices)

    @property
    def _view_graph(self):
        """ return a view of the graph """
        if not self._manager.graph:
            return None
        return self._pad_display_data(self._manager.graph_vertices)

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
        """ current time according to the first dimension """
        # TODO(arl): get the correct index here
        if isinstance(self.dims.indices[0], slice):
            return self._manager.max_time
        return self.dims.indices[0]

    @property
    def use_fade(self) -> bool:
        """ toggle whether we fade the tail of the track, depending on whether
        the time dimension is displayed """
        return 0 in self.dims.not_displayed

    @property
    def data(self) -> list:
        """list of (N, D) arrays: coordinates for N points in D dimensions."""
        return self._manager.data

    @data.setter
    def data(self, data: list):
        """ set the data and build the vispy arrays for display """
        self._manager.data = data

    @property
    def properties(self) -> list:
        """ return the list of track properties """
        return self._manager.properties

    @property
    def _property_keys(self):
        """ track properties that can be used for coloring etc... """
        return self._manager._property_keys

    @properties.setter
    def properties(self, properties: list):
        """ set track properties """
        self._manager.properties = properties
        print(self._manager.properties)
        # properties have been updated, we need to alert the gui
        self.events.properties()

    @property
    def graph(self) -> list:
        """ return the graph """
        return self._manager.graph

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
        """ display the track id """
        return self._display_id

    @display_id.setter
    def display_id(self, value: bool):
        self._display_id = value
        self.events.display_id()
        self.refresh()

    @property
    def display_tail(self) -> bool:
        """ display the track tail """
        return self._display_tail

    @display_tail.setter
    def display_tail(self, value: bool):
        self._display_tail = value
        self.events.display_tail()
        self.refresh()

    @property
    def display_graph(self) -> bool:
        """ display the graph edges """
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
        """ set the property to color vertices by """
        if color_by not in self._property_keys:
            return
        self._color_by = color_by
        self._recolor_tracks()
        # fire the events and update the display
        self.events.color_by()
        self.refresh()

    @property
    def colormap(self) -> str:
        return self._colormap

    @colormap.setter
    def colormap(self, colormap: str):
        """ set the default colormap """
        if colormap not in AVAILABLE_COLORMAPS:
            raise ValueError(f'Colormap {colormap} not available')
        self._colormap = colormap
        self._recolor_tracks()
        self.events.colormap()
        self.refresh()

    def _recolor_tracks(self):
        """ recolor the tracks """
        # if we change the coloring, rebuild the vertex colors array
        vertex_properties = self._manager.vertex_properties(self.color_by)

        def _norm(p):
            return (p - np.min(p)) / np.max([1e-10, np.ptp(p)])

        if self.color_by in self.colormaps_dict:
            colormap = self.colormaps_dict[self.color_by]
        else:
            # if we don't have a colormap, get one and scale the properties
            colormap = AVAILABLE_COLORMAPS[self.colormap]
            vertex_properties = _norm(vertex_properties)

        # actually set the vertex colors
        self._track_colors = colormap[vertex_properties]

    @property
    def track_connex(self) -> np.ndarray:
        """ vertex connections for drawing track lines """
        return self._manager.track_connex

    @property
    def track_colors(self) -> np.ndarray:
        """ return the vertex colors according to the currently selected
        property """
        return self._track_colors

    @property
    def graph_connex(self):
        """ vertex connections for drawing the graph """
        return self._manager.graph_connex

    @property
    def track_times(self) -> np.ndarray:
        """ time points associated with each track vertex """
        return self._manager.track_times

    @property
    def graph_times(self) -> np.ndarray:
        """ time points assocaite with each graph vertex """
        return self._manager.graph_times

    @property
    def track_labels(self) -> zip:
        """ return track labels at the current time """
        labels, positions = self._manager.track_labels(self.current_time)
        return zip(labels, self._pad_display_data(positions))
