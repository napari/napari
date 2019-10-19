from typing import Union
from xml.etree.ElementTree import Element
import numpy as np
import itertools
from copy import copy, deepcopy
from contextlib import contextmanager
from ..base import Layer
from ...util.event import Event
from ...util.misc import ensure_iterable
from ...util.status_messages import format_float
from vispy.color import get_color_names, Color
from ._constants import Symbol, SYMBOL_ALIAS, Mode


class Points(Layer):
    """Points layer.

    Parameters
    ----------
    data : array (N, D)
        Coordinates for N points in D dimensions.
    symbol : str
        Symbol to be used for the point markers. Must be one of the
        following: arrow, clobber, cross, diamond, disc, hbar, ring,
        square, star, tailed_arrow, triangle_down, triangle_up, vbar, x.
    size : float, array
        Size of the point marker. If given as a scalar, all points are made
        the same size. If given as an array, size must be the same
        broadcastable to the same shape as the data.
    edge_width : float
        Width of the symbol edge in pixels.
    edge_color : str
        Color of the point marker border.
    face_color : str
        Color of the point marker body.
    n_dimensional : bool
        If True, renders points not just in central plane but also in all
        n-dimensions according to specified point marker size.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.

    Attributes
    ----------
    data : array (N, D)
        Coordinates for N points in D dimensions.
    symbol : str
        Symbol used for all point markers.
    size : float
        Size of the marker for the next point to be added or the currently
        selected point.
    edge_width : float
        Width of the marker edges in pixels for all points
    edge_color : str
        Size of the marker edge for the next point to be added or the currently
        selected point.
    face_color : str
        Size of the marker edge for the next point to be added or the currently
        selected point.
    edge_colors : list of str (N,)
        List of edge color strings, one for each point.
    face_colors : list of str (N,)
        List of face color strings, one for each point.
    n_dimensional : bool
        If True, renders points not just in central plane but also in all
        n-dimensions according to specified point marker size.
    selected_data : list
        Integer indices of any selected points.
    sizes : array (N, D)
        Array of sizes for each point in each dimension. Must have the same
        shape as the layer `data`.
    mode : str
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In ADD mode clicks of the cursor add points at the clicked location.

        In SELECT mode the cursor can select points by clicking on them or
        by dragging a box around them. Once selected points can be moved,
        have their properties edited, or be deleted.

    Extended Summary
    ----------
    _data_view : array (M, 2)
        2D coordinates of points in the currently viewed slice.
    _sizes_view : array (M, )
        Size of the point markers in the currently viewed slice.
    _indices_view : array (M, )
        Integer indices of the points in the currently viewed slice.
    _selected_view :
        Integer indices of selected points in the currently viewed slice within
        the `_data_view` array.
    _selected_box : array (4, 2) or None
        Four corners of any box either around currently selected points or
        being created during a drag action. Starting in the top left and
        going clockwise.
    _drag_start : list or None
        Coordinates of first cursor click during a drag action. Gets reset to
        None after dragging is done.
    """

    # The max number of points that will ever be used to render the thumbnail
    # If more points are present then they are randomly subsampled
    _max_points_thumbnail = 1024

    def __init__(
        self,
        data=None,
        *,
        symbol='o',
        size=10,
        edge_width=1,
        edge_color='black',
        face_color='white',
        n_dimensional=False,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        opacity=1,
        blending='translucent',
        visible=True,
    ):
        if data is None:
            data = np.empty((0, 2))
        ndim = data.shape[1]
        super().__init__(
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
            mode=Event,
            size=Event,
            edge_width=Event,
            face_color=Event,
            edge_color=Event,
            symbol=Event,
            n_dimensional=Event,
            highlight=Event,
        )
        self._colors = get_color_names()

        # Save the point coordinates
        self._data = data
        self.dims.clip = False

        # Save the point style params
        self.symbol = symbol
        self._n_dimensional = n_dimensional
        self.edge_width = edge_width

        # The following point properties are for the new points that will
        # be added. For any given property, if a list is passed to the
        # constructor so each point gets its own value then the default
        # value is used when adding new points
        if np.isscalar(size):
            self._size = size
        else:
            self._size = 10

        if type(edge_color) is str:
            self._edge_color = edge_color
        else:
            self._edge_color = 'black'

        if type(face_color) is str:
            self._face_color = face_color
        else:
            self._face_color = 'white'

        # Indices of selected points
        self._selected_data = []
        self._selected_data_stored = []
        self._selected_data_history = []
        # Indices of selected points within the currently viewed slice
        self._selected_view = []
        # Index of hovered point
        self._value = None
        self._value_stored = None
        self._selected_box = None
        self._mode = Mode.PAN_ZOOM
        self._mode_history = self._mode
        self._status = self.mode

        self._drag_start = None

        # Nx2 array of points in the currently viewed slice
        self._data_view = np.empty((0, 2))
        # Sizes of points in the currently viewed slice
        self._sizes_view = 0
        # Full data indices of points located in the currently viewed slice
        self._indices_view = []

        self._drag_box = None
        self._drag_box_stored = None
        self._is_selecting = False
        self._clipboard = {}

        self.edge_colors = list(
            itertools.islice(
                ensure_iterable(edge_color, color=True), 0, len(self.data)
            )
        )
        self.face_colors = list(
            itertools.islice(
                ensure_iterable(face_color, color=True), 0, len(self.data)
            )
        )
        self.sizes = size

        # Trigger generation of view slice and thumbnail
        self._update_dims()

    @property
    def data(self) -> np.ndarray:
        """(N, D) array: coordinates for N points in D dimensions."""
        return self._data

    @data.setter
    def data(self, data: np.ndarray):
        cur_npoints = len(self._data)
        self._data = data

        # Adjust the size array when the number of points has changed
        if len(data) < cur_npoints:
            # If there are now less points, remove the sizes and colors of the
            # extra ones
            with self.events.set_data.blocker():
                self.edge_colors = self.edge_colors[: len(data)]
                self.face_colors = self.face_colors[: len(data)]
                self.sizes = self._sizes[: len(data)]

        elif len(data) > cur_npoints:
            # If there are now more points, add the sizes and colors of the
            # new ones
            with self.events.set_data.blocker():
                adding = len(data) - cur_npoints
                if len(self._sizes) > 0:
                    new_size = copy(self._sizes[-1])
                    for i in self.dims.displayed:
                        new_size[i] = self.size
                else:
                    # Add the default size, with a value for each dimension
                    new_size = np.repeat(self.size, self._sizes.shape[1])
                size = np.repeat([new_size], adding, axis=0)
                self.edge_colors += [self.edge_color for i in range(adding)]
                self.face_colors += [self.face_color for i in range(adding)]
                self.sizes = np.concatenate((self._sizes, size), axis=0)
        self._update_dims()
        self.events.data()

    def _get_ndim(self):
        """Determine number of dimensions of the layer."""
        return self.data.shape[1]

    def _get_extent(self):
        """Determine ranges for slicing given by (min, max, step)."""
        if len(self.data) == 0:
            maxs = np.ones(self.data.shape[1], dtype=int)
            mins = np.zeros(self.data.shape[1], dtype=int)
        else:
            maxs = np.max(self.data, axis=0)
            mins = np.min(self.data, axis=0)

        return [(min, max, 1) for min, max in zip(mins, maxs)]

    @property
    def n_dimensional(self) -> str:
        """bool: renders points as n-dimensionsal."""
        return self._n_dimensional

    @n_dimensional.setter
    def n_dimensional(self, n_dimensional: bool) -> None:
        self._n_dimensional = n_dimensional
        self.events.n_dimensional()
        self._set_view_slice()

    @property
    def symbol(self) -> str:
        """str: symbol used for all point markers."""
        return str(self._symbol)

    @symbol.setter
    def symbol(self, symbol: Union[str, Symbol]) -> None:

        if isinstance(symbol, str):
            # Convert the alias string to the deduplicated string
            if symbol in SYMBOL_ALIAS:
                symbol = SYMBOL_ALIAS[symbol]
            else:
                symbol = Symbol(symbol)
        self._symbol = symbol
        self.events.symbol()
        self.events.highlight()

    @property
    def sizes(self) -> Union[int, float, np.ndarray, list]:
        """(N, D) array: sizes of all N points in D dimensions."""
        return self._sizes

    @sizes.setter
    def sizes(self, size: Union[int, float, np.ndarray, list]) -> None:
        try:
            self._sizes = np.broadcast_to(size, self.data.shape).copy()
        except:
            try:
                self._sizes = np.broadcast_to(
                    size, self.data.shape[::-1]
                ).T.copy()
            except:
                raise ValueError("Size is not compatible for broadcasting")
        self._set_view_slice()

    @property
    def size(self) -> Union[int, float]:
        """float: size of marker for the next added point."""
        return self._size

    @size.setter
    def size(self, size: Union[None, float]) -> None:
        self._size = size
        if self._update_properties and len(self.selected_data) > 0:
            for i in self.selected_data:
                self.sizes[i, :] = (self.sizes[i, :] > 0) * size
            self._set_view_slice()
        self.status = format_float(self.size)
        self.events.size()

    @property
    def edge_width(self) -> Union[None, int, float]:
        """float: width used for all point markers."""

        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: Union[None, float]) -> None:
        self._edge_width = edge_width
        self.status = format_float(self.edge_width)
        self.events.edge_width()
        self.events.highlight()

    @property
    def edge_color(self) -> str:
        """str: edge color of marker for the next added point."""

        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str) -> None:
        self._edge_color = edge_color
        if self._update_properties and len(self.selected_data) > 0:
            for i in self.selected_data:
                self.edge_colors[i] = edge_color
        self.events.edge_color()
        self.events.highlight()

    @property
    def face_color(self) -> str:
        """str: face color of marker for the next added point."""

        return self._face_color

    @face_color.setter
    def face_color(self, face_color: str) -> None:
        self._face_color = face_color
        if self._update_properties and len(self.selected_data) > 0:
            for i in self.selected_data:
                self.face_colors[i] = face_color
        self.events.face_color()
        self.events.highlight()

    @property
    def selected_data(self):
        """list: list of currently selected points."""
        return self._selected_data

    @selected_data.setter
    def selected_data(self, selected_data):
        self._selected_data = list(selected_data)
        selected = []
        for c in self._selected_data:
            if c in self._indices_view:
                ind = list(self._indices_view).index(c)
                selected.append(ind)
        self._selected_view = selected
        self._selected_box = self.interaction_box(self._selected_view)

        # Update properties based on selected points
        index = self._selected_data
        edge_colors = list(set([self.edge_colors[i] for i in index]))
        if len(edge_colors) == 1:
            edge_color = edge_colors[0]
            with self.block_update_properties():
                self.edge_color = edge_color

        face_colors = list(set([self.face_colors[i] for i in index]))
        if len(face_colors) == 1:
            face_color = face_colors[0]
            with self.block_update_properties():
                self.face_color = face_color

        size = list(
            set([self.sizes[i, self.dims.displayed].mean() for i in index])
        )
        if len(size) == 1:
            size = size[0]
            with self.block_update_properties():
                self.size = size

    def interaction_box(self, index):
        """Create the interaction box around a list of points in view.

        Parameters
        ----------
        index : list
            List of points around which to construct the interaction box.

        Returns
        ----------
        box : np.ndarray
            4x2 array of corners of the interaction box in clockwise order
            starting in the upper-left corner.
        """
        if len(index) == 0:
            box = None
        else:
            data = self._data_view[index]
            size = self._sizes_view[index]
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            data = points_to_squares(data, size)
            box = create_box(data)

        return box

    @property
    def mode(self):
        """str: Interactive mode

        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In ADD mode clicks of the cursor add points at the clicked location.

        In SELECT mode the cursor can select points by clicking on them or
        by dragging a box around them. Once selected points can be moved,
        have their properties edited, or be deleted.
        """
        return str(self._mode)

    @mode.setter
    def mode(self, mode):
        if isinstance(mode, str):
            mode = Mode(mode)

        if not self.editable:
            mode = Mode.PAN_ZOOM

        if mode == self._mode:
            return
        old_mode = self._mode

        if mode == Mode.ADD:
            self.cursor = 'pointing'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom'
        elif mode == Mode.SELECT:
            self.cursor = 'standard'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom'
        elif mode == Mode.PAN_ZOOM:
            self.cursor = 'standard'
            self.interactive = True
            self.help = ''
        else:
            raise ValueError("Mode not recognized")

        if not (mode == Mode.SELECT and old_mode == Mode.SELECT):
            self.selected_data = []
            self._set_highlight()

        self.status = str(mode)
        self._mode = mode

        self.events.mode(mode=mode)

    def _set_editable(self, editable=None):
        """Set editable mode based on layer properties."""
        if editable is None:
            if self.dims.ndisplay == 3:
                self.editable = False
            else:
                self.editable = True

        if self.editable == False:
            self.mode = Mode.PAN_ZOOM

    def _slice_data(self, indices):
        """Determines the slice of points given the indices.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.

        Returns
        ----------
        in_slice_data : (N, 2) array
            Coordinates of points in the currently viewed slice.
        slice_indices : list
            Indices of points in the currently viewed slice.
        scale : float, (N, ) array
            If in `n_dimensional` mode then the scale factor of points, where
            values of 1 corresponds to points located in the slice, and values
            less than 1 correspond to points located in neighboring slices.
        """
        # Get a list of the data for the points in this slice
        not_disp = list(self.dims.not_displayed)
        disp = list(self.dims.displayed)
        indices = np.array(indices)
        if len(self.data) > 0:
            if self.n_dimensional is True and self.ndim > 2:
                distances = abs(self.data[:, not_disp] - indices[not_disp])
                sizes = self.sizes[:, not_disp] / 2
                matches = np.all(distances <= sizes, axis=1)
                in_slice_data = self.data[np.ix_(matches, disp)]
                size_match = sizes[matches]
                size_match[size_match == 0] = 1
                scale_per_dim = (size_match - distances[matches]) / size_match
                scale_per_dim[size_match == 0] = 1
                scale = np.prod(scale_per_dim, axis=1)
                indices = np.where(matches)[0].astype(int)
                return in_slice_data, indices, scale
            else:
                data = self.data[:, not_disp].astype('int')
                matches = np.all(data == indices[not_disp], axis=1)
                in_slice_data = self.data[np.ix_(matches, disp)]
                indices = np.where(matches)[0].astype(int)
                return in_slice_data, indices, 1
        else:
            return [], [], []

    def get_value(self):
        """Determine if points at current coordinates.

        Returns
        ----------
        selection : int or None
            Index of point that is at the current coordinate if any.
        """
        in_slice_data = self._data_view

        # Display points if there are any in this slice
        if len(self._data_view) > 0:
            # Get the point sizes
            distances = abs(
                self._data_view
                - [self.coordinates[d] for d in self.dims.displayed]
            )
            in_slice_matches = np.all(
                distances <= np.expand_dims(self._sizes_view, axis=1) / 2,
                axis=1,
            )
            indices = np.where(in_slice_matches)[0]
            if len(indices) > 0:
                selection = self._indices_view[indices[-1]]
            else:
                selection = None
        else:
            selection = None

        return selection

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""

        in_slice_data, indices, scale = self._slice_data(self.dims.indices)

        # Display points if there are any in this slice
        if len(in_slice_data) > 0:
            # Get the point sizes
            sizes = (
                self.sizes[np.ix_(indices, self.dims.displayed)].mean(axis=1)
                * scale
            )

            # Update the points node
            data = np.array(in_slice_data)

        else:
            # if no points in this slice send dummy data
            data = np.zeros((0, self.dims.ndisplay))
            sizes = [0]
        self._data_view = data
        self._sizes_view = sizes
        self._indices_view = indices
        # Make sure if changing planes any selected points not in the current
        # plane are removed
        selected = []
        for c in self.selected_data:
            if c in self._indices_view:
                ind = list(self._indices_view).index(c)
                selected.append(ind)
        self._selected_view = selected
        if len(selected) == 0:
            self.selected_data
        self._selected_box = self.interaction_box(self._selected_view)

        self._set_highlight(force=True)
        self._update_thumbnail()
        self._update_coordinates()
        self.events.set_data()

    def _set_highlight(self, force=False):
        """Render highlights of shapes including boundaries, vertices,
        interaction boxes, and the drag selection box when appropriate

        Parameters
        ----------
        force : bool
            Bool that forces a redraw to occur when `True`
        """
        # Check if any point ids have changed since last call
        if (
            self.selected_data == self._selected_data_stored
            and self._value == self._value_stored
            and np.all(self._drag_box == self._drag_box_stored)
        ) and not force:
            return
        self._selected_data_stored = copy(self.selected_data)
        self._value_stored = copy(self._value)
        self._drag_box_stored = copy(self._drag_box)

        if self._mode == Mode.SELECT and (
            self._value is not None or len(self._selected_view) > 0
        ):
            if len(self._selected_view) > 0:
                index = copy(self._selected_view)
                if self._value is not None:
                    hover_point = list(self._indices_view).index(self._value)
                    if hover_point in index:
                        pass
                    else:
                        index.append(hover_point)
                index.sort()
            else:
                hover_point = list(self._indices_view).index(self._value)
                index = [hover_point]

            self._highlight_index = index
        else:
            self._highlight_index = []

        pos = self._selected_box
        if pos is None and not self._is_selecting:
            pos = np.zeros((0, 2))
        elif self._is_selecting:
            pos = create_box(self._drag_box)
            pos = pos[list(range(4)) + [0]]
        else:
            pos = pos[list(range(4)) + [0]]

        self._highlight_box = pos
        self.events.highlight()

    def _update_thumbnail(self):
        """Update thumbnail with current points and colors."""
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        if len(self._data_view) > 0:
            min_vals = [self.dims.range[i][0] for i in self.dims.displayed]
            shape = np.ceil(
                [
                    self.dims.range[i][1] - self.dims.range[i][0] + 1
                    for i in self.dims.displayed
                ]
            ).astype(int)
            zoom_factor = np.divide(
                self._thumbnail_shape[:2], shape[-2:]
            ).min()
            if len(self._data_view) > self._max_points_thumbnail:
                inds = np.random.randint(
                    0, len(self._data_view), self._max_points_thumbnail
                )
                points = self._data_view[inds]
            else:
                points = self._data_view
            coords = np.floor(
                (points[:, -2:] - min_vals[-2:] + 0.5) * zoom_factor
            ).astype(int)
            coords = np.clip(
                coords, 0, np.subtract(self._thumbnail_shape[:2], 1)
            )
            for i, c in enumerate(coords):
                col = self.face_colors[self._indices_view[i]]
                colormapped[c[0], c[1], :] = Color(col).rgba
        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    def add(self, coord):
        """Adds point at coordinate.

        Parameters
        ----------
        coord : sequence of indices to add point at
        """
        self.data = np.append(self.data, [coord], axis=0)

    def remove_selected(self):
        """Removes selected points if any."""
        index = copy(self.selected_data)
        index.sort()
        if len(index) > 0:
            self._sizes = np.delete(self._sizes, index, axis=0)
            for i in index[::-1]:
                del self.edge_colors[i]
                del self.face_colors[i]
            if self._value in self.selected_data:
                self._value = None
            self.selected_data = []
            self.data = np.delete(self.data, index, axis=0)

    def _move(self, index, coord):
        """Moves points relative drag start location.

        Parameters
        ----------
        index : list
            Integer indices of points to move
        coord : tuple
            Coordinates to move points to
        """
        if len(index) > 0:
            disp = list(self.dims.displayed)
            if self._drag_start is None:
                center = self.data[np.ix_(index, disp)].mean(axis=0)
                self._drag_start = np.array(coord)[disp] - center
            center = self.data[np.ix_(index, disp)].mean(axis=0)
            shift = np.array(coord)[disp] - center - self._drag_start
            self.data[np.ix_(index, disp)] = (
                self.data[np.ix_(index, disp)] + shift
            )
            self._set_view_slice()

    def _copy_data(self):
        """Copy selected points to clipboard."""
        if len(self.selected_data) > 0:
            self._clipboard = {
                'data': deepcopy(self.data[self.selected_data]),
                'size': deepcopy(self.sizes[self.selected_data]),
                'edge_color': deepcopy(
                    [self.edge_colors[i] for i in self.selected_data]
                ),
                'face_color': deepcopy(
                    [self.face_colors[i] for i in self.selected_data]
                ),
                'indices': self.dims.indices,
            }
        else:
            self._clipboard = {}

    def _paste_data(self):
        """Paste any point from clipboard and select them."""
        npoints = len(self._data_view)
        totpoints = len(self.data)

        if len(self._clipboard.keys()) > 0:
            not_disp = self.dims.not_displayed
            data = deepcopy(self._clipboard['data'])
            offset = [
                self.dims.indices[i] - self._clipboard['indices'][i]
                for i in not_disp
            ]
            data[:, not_disp] = data[:, not_disp] + np.array(offset)
            self._data = np.append(self.data, data, axis=0)
            self._sizes = np.append(
                self.sizes, deepcopy(self._clipboard['size']), axis=0
            )
            self.edge_colors = self.edge_colors + deepcopy(
                self._clipboard['edge_color']
            )
            self.face_colors = self.face_colors + deepcopy(
                self._clipboard['face_color']
            )
            self._selected_view = list(
                range(npoints, npoints + len(self._clipboard['data']))
            )
            self._selected_data = list(
                range(totpoints, totpoints + len(self._clipboard['data']))
            )
            self._set_view_slice()

    def to_xml_list(self):
        """Convert the points to a list of xml elements according to the svg
        specification. Z ordering of the points will be taken into account.
        Each point is represented by a circle. Support for other symbols is
        not yet implemented.

        Returns
        ----------
        xml : list
            List of xml elements defining each point according to the
            svg specification
        """
        xml_list = []
        width = str(self.edge_width)
        opacity = str(self.opacity)
        props = {'stroke-width': width, 'opacity': opacity}

        for i, d, s in zip(
            self._indices_view, self._data_view, self._sizes_view
        ):
            d = d[::-1]
            cx = str(d[0])
            cy = str(d[1])
            r = str(s / 2)
            face_color = (255 * Color(self.face_colors[i]).rgba).astype(np.int)
            fill = f'rgb{tuple(face_color[:3])}'
            edge_color = (255 * Color(self.edge_colors[i]).rgba).astype(np.int)
            stroke = f'rgb{tuple(edge_color[:3])}'

            element = Element(
                'circle', cx=cx, cy=cy, r=r, stroke=stroke, fill=fill, **props
            )
            xml_list.append(element)

        return xml_list

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        if self._mode == Mode.SELECT:
            if event.is_dragging:
                if len(self.selected_data) > 0:
                    self._move(self.selected_data, self.coordinates)
                else:
                    self._is_selecting = True
                    if self._drag_start is None:
                        self._drag_start = [
                            self.coordinates[d] for d in self.dims.displayed
                        ]
                    self._drag_box = np.array(
                        [
                            self._drag_start,
                            [self.coordinates[d] for d in self.dims.displayed],
                        ]
                    )
                    self._set_highlight()
            else:
                self._set_highlight()

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.
        """
        shift = 'Shift' in event.modifiers

        if self._mode == Mode.SELECT:
            if shift and self._value is not None:
                if self._value in self.selected_data:
                    self.selected_data = [
                        x for x in self.selected_data if x != self._value
                    ]
                else:
                    self.selected_data += [self._value]
            elif self._value is not None:
                if self._value not in self.selected_data:
                    self.selected_data = [self._value]
            else:
                self.selected_data = []
            self._set_highlight()
        elif self._mode == Mode.ADD:
            self.add(self.coordinates)

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.
        """
        self._drag_start = None
        if self._is_selecting:
            self._is_selecting = False
            if len(self._data_view) > 0:
                selection = points_in_box(
                    self._drag_box, self._data_view, self._sizes_view
                )
                self.selected_data = self._indices_view[selection]
            else:
                self.selected_data = []
            self._set_highlight(force=True)


def create_box(data):
    """Create the axis aligned interaction box of a list of points

    Parameters
    ----------
    data : (N, 2) array
        Points around which the interaction box is created

    Returns
    -------
    box : (4, 2) array
        Vertices of the interaction box
    """
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    tl = np.array([min_val[0], min_val[1]])
    tr = np.array([max_val[0], min_val[1]])
    br = np.array([max_val[0], max_val[1]])
    bl = np.array([min_val[0], max_val[1]])
    box = np.array([tl, tr, br, bl])
    return box


def points_to_squares(points, sizes):
    """Expand points to squares defined by their size

    Parameters
    ----------
    points : (N, 2) array
        Points to be turned into squares
    sizes : (N,) array
        Size of each point

    Returns
    -------
    rect : (4N, 2) array
        Vertices of the expanded points
    """
    rect = np.concatenate(
        [
            points + np.sqrt(2) / 2 * np.array([sizes, sizes]).T,
            points + np.sqrt(2) / 2 * np.array([sizes, -sizes]).T,
            points + np.sqrt(2) / 2 * np.array([-sizes, sizes]).T,
            points + np.sqrt(2) / 2 * np.array([-sizes, -sizes]).T,
        ],
        axis=0,
    )
    return rect


def points_in_box(corners, points, sizes):
    """Determine which points are in an axis aligned box defined by the corners

    Parameters
    ----------
    points : (N, 2) array
        Points to be checked
    sizes : (N,) array
        Size of each point

    Returns
    -------
    inside : list
        Indices of points inside the box
    """
    box = create_box(corners)[[0, 2]]
    rect = points_to_squares(points, sizes)
    below_top = np.all(box[1] >= rect, axis=1)
    above_bottom = np.all(rect >= box[0], axis=1)
    inside = np.logical_and(below_top, above_bottom)
    inside = np.unique(np.where(inside)[0] % len(points))
    return list(inside)
