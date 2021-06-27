# from napari.layers.base.base import Layer
# from napari.utils.events import Event
# from napari.utils.colormaps import AVAILABLE_COLORMAPS

from typing import Dict, List, Union
from warnings import warn

import numpy as np

from ...utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from ...utils.events import Event
from ...utils.translations import trans
from ..base import Layer
from ._track_utils import TrackManager


class Tracks(Layer):
    """Tracks layer.

    Parameters
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
    color_by: str
        Track property (from property keys) by which to color vertices.
    tail_width : float
        Width of the track tails in pixels.
    tail_length : float
        Length of the track tails in units of time.
    colormap : str
        Default colormap to use to set vertex colors. Specialized colormaps,
        relating to specified properties can be passed to the layer via
        colormaps_dict.
    colormaps_dict : dict {str: napari.utils.Colormap}
        Optional dictionary mapping each property to a colormap for that
        property. This allows each property to be assigned a specific colormap,
        rather than having a global colormap for everything.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a lenght N translation vector and a 1 or a napari
        AffineTransform object. If provided then translate, scale, rotate, and
        shear values are ignored.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.


    """

    # The max number of tracks that will ever be used to render the thumbnail
    # If more tracks are present then they are randomly subsampled
    _max_tracks_thumbnail = 1024

    def __init__(
        self,
        data,
        *,
        properties=None,
        graph=None,
        tail_width=2,
        tail_length=30,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1,
        blending='additive',
        visible=True,
        colormap='turbo',
        color_by='track_id',
        colormaps_dict=None,
    ):

        # if not provided with any data, set up an empty layer in 2D+t
        if data is None:
            data = np.empty((0, 4))
        else:
            # convert data to a numpy array if it is not already one
            data = np.asarray(data)

        # in absence of properties make the default an empty dict
        if properties is None:
            properties = {}

        # set the track data dimensions (remove ID from data)
        ndim = data.shape[1] - 1

        super().__init__(
            data,
            ndim,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            rotate=rotate,
            shear=shear,
            affine=affine,
            opacity=opacity,
            blending=blending,
            visible=visible,
        )

        self.events.add(
            tail_width=Event,
            tail_length=Event,
            display_id=Event,
            display_tail=Event,
            display_graph=Event,
            color_by=Event,
            colormap=Event,
            properties=Event,
            rebuild_tracks=Event,
            rebuild_graph=Event,
        )

        # track manager deals with data slicing, graph building and properties
        self._manager = TrackManager()
        self._track_colors = None
        self._colormaps_dict = colormaps_dict or {}  # additional colormaps
        self._color_by = color_by  # default color by ID
        self._colormap = colormap

        # use this to update shaders when the displayed dims change
        self._current_displayed_dims = None

        # track display properties
        self.tail_width = tail_width
        self.tail_length = tail_length
        self.display_id = False
        self.display_tail = True
        self.display_graph = True

        # set the data, properties and graph
        self.data = data
        self.properties = properties
        self.graph = graph or {}

        self.color_by = color_by
        self.colormap = colormap

        self._update_dims()

        # reset the display before returning
        self._current_displayed_dims = None

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        if len(self.data) == 0:
            extrema = np.full((2, self.ndim), np.nan)
        else:
            maxs = np.max(self.data, axis=0)
            mins = np.min(self.data, axis=0)
            extrema = np.vstack([mins, maxs])
        return extrema[:, 1:]

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
                'data': self.data,
                'properties': self.properties,
                'graph': self.graph,
                'color_by': self.color_by,
                'colormap': self.colormap,
                'colormaps_dict': self.colormaps_dict,
                'tail_width': self.tail_width,
                'tail_length': self.tail_length,
            }
        )
        return state

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""

        # if the displayed dims have changed, update the shader data
        if self._dims_displayed != self._current_displayed_dims:
            # store the new dims
            self._current_displayed_dims = self._dims_displayed
            # fire the events to update the shaders
            self.events.rebuild_tracks()
            self.events.rebuild_graph()

        return

    def _get_value(self, position) -> int:
        """Value of the data at a position in data coordinates.

        Use a kd-tree to lookup the ID of the nearest tree.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : int or None
            Index of track that is at the current coordinate if any.
        """
        return self._manager.get_value(np.array(position))

    def _update_thumbnail(self):
        """Update thumbnail with current points and colors."""
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1

        if self._view_data is not None and self.track_colors is not None:
            de = self._extent_data
            min_vals = [de[0, i] for i in self._dims_displayed]
            shape = np.ceil(
                [de[1, i] - de[0, i] + 1 for i in self._dims_displayed]
            ).astype(int)
            zoom_factor = np.divide(
                self._thumbnail_shape[:2], shape[-2:]
            ).min()
            if len(self._view_data) > self._max_tracks_thumbnail:
                thumbnail_indices = np.random.randint(
                    0, len(self._view_data), self._max_tracks_thumbnail
                )
                points = self._view_data[thumbnail_indices]
            else:
                points = self._view_data
                thumbnail_indices = range(len(self._view_data))

            # get the track coords here
            coords = np.floor(
                (points[:, :2] - min_vals[1:] + 0.5) * zoom_factor
            ).astype(int)
            coords = np.clip(
                coords, 0, np.subtract(self._thumbnail_shape[:2], 1)
            )

            # modulate track colors as per colormap/current_time
            colors = self.track_colors[thumbnail_indices]
            times = self.track_times[thumbnail_indices]
            alpha = (self.current_time - times) / self.tail_length
            alpha[times > self.current_time] = 1.0
            colors[:, -1] = np.clip(1.0 - alpha, 0.0, 1.0)
            colormapped[coords[:, 1], coords[:, 0]] = colors

        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    @property
    def _view_data(self):
        """return a view of the data"""
        return self._pad_display_data(self._manager.track_vertices)

    @property
    def _view_graph(self):
        """return a view of the graph"""
        return self._pad_display_data(self._manager.graph_vertices)

    def _pad_display_data(self, vertices):
        """pad display data when moving between 2d and 3d"""
        if vertices is None:
            return

        data = vertices[:, self._dims_displayed]
        # if we're only displaying two dimensions, then pad the display dim
        # with zeros
        if self._ndisplay == 2:
            data = np.pad(data, ((0, 0), (0, 1)), 'constant')
            return data[:, (1, 0, 2)]  # y, x, z -> x, y, z
        else:
            return data[:, (2, 1, 0)]  # z, y, x -> x, y, z

    @property
    def current_time(self):
        """current time according to the first dimension"""
        # TODO(arl): get the correct index here
        time_step = self._slice_indices[0]

        if isinstance(time_step, slice):
            # if we are visualizing all time, then just set to the maximum
            # timestamp of the dataset
            return self._manager.max_time

        return time_step

    @property
    def use_fade(self) -> bool:
        """toggle whether we fade the tail of the track, depending on whether
        the time dimension is displayed"""
        return 0 in self._dims_not_displayed

    @property
    def data(self) -> np.ndarray:
        """array (N, D+1): Coordinates for N points in D+1 dimensions."""
        return self._manager.data

    @data.setter
    def data(self, data: np.ndarray):
        """set the data and build the vispy arrays for display"""
        # set the data and build the tracks
        self._manager.data = data
        self._manager.build_tracks()

        # reset the properties and recolor the tracks
        self.properties = {}
        self._recolor_tracks()

        # reset the graph
        self._manager.graph = {}
        self._manager.build_graph()

        # fire events to update shaders
        self.events.rebuild_tracks()
        self.events.rebuild_graph()
        self.events.data(value=self.data)
        self._set_editable()
        self._update_dims()

    @property
    def properties(self) -> Dict[str, np.ndarray]:
        """dict {str: np.ndarray (N,)}, DataFrame: Properties for each track."""
        return self._manager.properties

    @property
    def properties_to_color_by(self) -> List[str]:
        """track properties that can be used for coloring etc..."""
        return list(self.properties.keys())

    @properties.setter
    def properties(self, properties: Dict[str, np.ndarray]):
        """set track properties"""
        if self._color_by not in [*properties.keys(), 'track_id']:
            warn(
                (
                    trans._(
                        "Previous color_by key {key!r} not present in new properties. Falling back to track_id",
                        deferred=True,
                        key=self._color_by,
                    )
                ),
                UserWarning,
            )
            self._color_by = 'track_id'
        self._manager.properties = properties
        self.events.properties()
        self.events.color_by()

    @property
    def graph(self) -> Dict[int, Union[int, List[int]]]:
        """dict {int: list}: Graph representing associations between tracks."""
        return self._manager.graph

    @graph.setter
    def graph(self, graph: Dict[int, Union[int, List[int]]]):
        """Set the track graph."""
        self._manager.graph = graph
        self._manager.build_graph()
        self.events.rebuild_graph()

    @property
    def tail_width(self) -> Union[int, float]:
        """float: Width for all vectors in pixels."""
        return self._tail_width

    @tail_width.setter
    def tail_width(self, tail_width: Union[int, float]):
        self._tail_width = tail_width
        self.events.tail_width()

    @property
    def tail_length(self) -> Union[int, float]:
        """float: Width for all vectors in pixels."""
        return self._tail_length

    @tail_length.setter
    def tail_length(self, tail_length: Union[int, float]):
        self._tail_length = tail_length
        self.events.tail_length()

    @property
    def display_id(self) -> bool:
        """display the track id"""
        return self._display_id

    @display_id.setter
    def display_id(self, value: bool):
        self._display_id = value
        self.events.display_id()
        self.refresh()

    @property
    def display_tail(self) -> bool:
        """display the track tail"""
        return self._display_tail

    @display_tail.setter
    def display_tail(self, value: bool):
        self._display_tail = value
        self.events.display_tail()

    @property
    def display_graph(self) -> bool:
        """display the graph edges"""
        return self._display_graph

    @display_graph.setter
    def display_graph(self, value: bool):
        self._display_graph = value
        self.events.display_graph()

    @property
    def color_by(self) -> str:
        return self._color_by

    @color_by.setter
    def color_by(self, color_by: str):
        """set the property to color vertices by"""
        if color_by not in self.properties_to_color_by:
            raise ValueError(
                trans._(
                    '{color_by} is not a valid property key',
                    deferred=True,
                    color_by=color_by,
                )
            )
        self._color_by = color_by
        self._recolor_tracks()
        self.events.color_by()

    @property
    def colormap(self) -> str:
        return self._colormap

    @colormap.setter
    def colormap(self, colormap: str):
        """set the default colormap"""
        if colormap not in AVAILABLE_COLORMAPS:
            raise ValueError(
                trans._(
                    'Colormap {colormap} not available',
                    deferred=True,
                    colormap=colormap,
                )
            )
        self._colormap = colormap
        self._recolor_tracks()
        self.events.colormap()

    @property
    def colormaps_dict(self) -> Dict[str, Colormap]:
        return self._colormaps_dict

    @colormaps_dict.setter
    def colomaps_dict(self, colormaps_dict: Dict[str, Colormap]):
        # validate the dictionary entries?
        self._colormaps_dict = colormaps_dict

    def _recolor_tracks(self):
        """recolor the tracks"""

        # this catch prevents a problem coloring the tracks if the data is
        # updated before the properties are. properties should always contain
        # a track_id key
        if self.color_by not in self.properties_to_color_by:
            self._color_by = 'track_id'
            self.events.color_by()

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
        self._track_colors = colormap.map(vertex_properties)

    @property
    def track_connex(self) -> np.ndarray:
        """vertex connections for drawing track lines"""
        return self._manager.track_connex

    @property
    def track_colors(self) -> np.ndarray:
        """return the vertex colors according to the currently selected
        property"""
        return self._track_colors

    @property
    def graph_connex(self) -> np.ndarray:
        """vertex connections for drawing the graph"""
        return self._manager.graph_connex

    @property
    def track_times(self) -> np.ndarray:
        """time points associated with each track vertex"""
        return self._manager.track_times

    @property
    def graph_times(self) -> np.ndarray:
        """time points assocaite with each graph vertex"""
        return self._manager.graph_times

    @property
    def track_labels(self) -> tuple:
        """return track labels at the current time"""
        labels, positions = self._manager.track_labels(self.current_time)

        # if there are no labels, return empty for vispy
        if not labels:
            return None, (None, None)

        padded_positions = self._pad_display_data(positions)
        return labels, padded_positions
