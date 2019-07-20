from typing import Union
from xml.etree.ElementTree import Element
import numpy as np
import itertools
from copy import copy, deepcopy
from contextlib import contextmanager
from ..base import Layer
from vispy.scene.visuals import Line, Markers, Compound
from ...util.event import Event
from ...util.misc import ensure_iterable
from vispy.color import get_color_names, Color
from ._constants import Symbol, SYMBOL_ALIAS, Mode


class Points(Layer):
    """Points layer.

    Parameters
    ----------
    coords : array (N, D)
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

    _highlight_color = (0, 0.6, 1)
    _highlight_width = 1.5

    class_keymap = {}

    def __init__(
        self,
        coords,
        *,
        symbol='o',
        size=10,
        edge_width=1,
        edge_color='black',
        face_color='white',
        n_dimensional=False,
        name=None,
    ):

        # Create a compound visual with the following four subvisuals:
        # Lines: The lines of the interaction box used for highlights.
        # Markers: The the outlines for each point used for highlights.
        # Markers: The actual markers of each point.
        visual = Compound([Line(), Markers(), Markers()])

        super().__init__(visual, name)

        self.events.add(
            mode=Event,
            size=Event,
            face_color=Event,
            edge_color=Event,
            symbol=Event,
            n_dimensional=Event,
        )
        self._colors = get_color_names()

        # Freeze refreshes
        with self.freeze_refresh():
            # Save the point coordinates
            self._data = coords

            # Save the point style params
            self.symbol = symbol
            self.n_dimensional = n_dimensional
            self.edge_width = edge_width

            self.sizes = size
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
            self._hover_point = None
            self._hover_point_stored = None
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

            # update flags
            self._need_display_update = False
            self._need_visual_update = False

            # Re intitialize indices depending on image dims
            self._indices = (0,) * (self.ndim - 2) + (
                slice(None, None, None),
                slice(None, None, None),
            )

            # Trigger generation of view slice and thumbnail
            self._set_view_slice()

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
            with self.freeze_refresh():
                self.sizes = self._sizes[: len(data)]
                self.edge_colors = self.edge_colors[: len(data)]
                self.face_colors = self.face_colors[: len(data)]

        elif len(data) > cur_npoints:
            # If there are now more points, add the sizes and colors of the
            # new ones
            with self.freeze_refresh():
                adding = len(data) - cur_npoints
                if len(self._sizes) > 0:
                    new_size = copy(self._sizes[-1])
                    new_size[-2:] = self.size
                else:
                    # Add the default size, with a value for each dimension
                    new_size = np.repeat(self.size, self._sizes.shape[1])
                size = np.repeat([new_size], adding, axis=0)
                self.sizes = np.concatenate((self._sizes, size), axis=0)
                self.edge_colors += [self.edge_color for i in range(adding)]
                self.face_colors += [self.face_color for i in range(adding)]

        self.events.data()
        self.refresh()

    @property
    def n_dimensional(self) -> str:
        """bool: renders points as n-dimensionsal."""
        return self._n_dimensional

    @n_dimensional.setter
    def n_dimensional(self, n_dimensional: bool) -> None:
        self._n_dimensional = n_dimensional
        self.events.n_dimensional()

        self.refresh()

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

        self.refresh()

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
        self.refresh()

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
            self.refresh()
        self.events.size()

    @property
    def edge_width(self) -> Union[None, int, float]:
        """float: width used for all point markers."""

        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: Union[None, float]) -> None:
        self._edge_width = edge_width

        self.refresh()

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
            self.refresh()
        self.events.edge_color()

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
            self.refresh()
        self.events.face_color()

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

        size = list(set([self.sizes[i, -2:].mean() for i in index]))
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

    def _get_shape(self):
        if len(self.data) == 0:
            # when no points given, return (1,) * dim
            # we use data.shape[1] as the dimensionality of the image
            return np.ones(self.data.shape[1], dtype=int)
        else:
            return np.max(self.data, axis=0)

    @property
    def range(self):
        """list of 3-tuple: ranges for slicing given by (min, max, step)."""
        if len(self.data) == 0:
            maxs = np.ones(self.data.shape[1], dtype=int)
            mins = np.zeros(self.data.shape[1], dtype=int)
        else:
            maxs = np.max(self.data, axis=0)
            mins = np.min(self.data, axis=0)

        return [(min, max, 1) for min, max in zip(mins, maxs)]

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
        if len(self.data) > 0:
            if self.n_dimensional is True and self.ndim > 2:
                distances = abs(self.data[:, :-2] - indices[:-2])
                sizes = self.sizes[:, :-2] / 2
                matches = np.all(distances <= sizes, axis=1)
                in_slice_data = self.data[matches, -2:]
                size_match = sizes[matches]
                size_match[size_match == 0] = 1
                scale_per_dim = (size_match - distances[matches]) / size_match
                scale_per_dim[size_match == 0] = 1
                scale = np.prod(scale_per_dim, axis=1)
                indices = np.where(matches)[0].astype(int)
                return in_slice_data, indices, scale
            else:
                matches = np.all(self.data[:, :-2] == indices[:-2], axis=1)
                in_slice_data = self.data[matches, -2:]
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
            distances = abs(self._data_view - self.coordinates[-2:])
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

        in_slice_data, indices, scale = self._slice_data(self.indices)

        # Display points if there are any in this slice
        if len(in_slice_data) > 0:
            # Get the point sizes
            sizes = self.sizes[indices, -2:].mean(axis=1) * scale

            # Update the points node
            data = np.array(in_slice_data) + 0.5

        else:
            # if no points in this slice send dummy data
            data = np.empty((0, 2))
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

        if len(data) > 0:
            edge_color = [
                self.edge_colors[i] for i in self._indices_view[::-1]
            ]
            face_color = [
                self.face_colors[i] for i in self._indices_view[::-1]
            ]
        else:
            edge_color = 'white'
            face_color = 'white'

        # Set vispy data, noting that the order of the points needs to be
        # reversed to make the most recently added point appear on top
        # and the rows / columns need to be switch for vispys x / y ordering
        self._node._subvisuals[2].set_data(
            data[::-1, [1, 0]],
            size=sizes[::-1],
            edge_width=self.edge_width,
            symbol=self.symbol,
            edge_color=edge_color,
            face_color=face_color,
            scaling=True,
        )
        self._need_visual_update = True
        self._set_highlight(force=True)
        self._update()
        self.status = self.get_message(self.coordinates, self._hover_point)
        self._update_thumbnail()

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
            and self._hover_point == self._hover_point_stored
            and np.all(self._drag_box == self._drag_box_stored)
        ) and not force:
            return
        self._selected_data_stored = copy(self.selected_data)
        self._hover_point_stored = copy(self._hover_point)
        self._drag_box_stored = copy(self._drag_box)

        if self._mode == Mode.SELECT and (
            self._hover_point is not None or len(self._selected_view) > 0
        ):
            if len(self._selected_view) > 0:
                index = copy(self._selected_view)
                if self._hover_point is not None:
                    hover_point = list(self._indices_view).index(
                        self._hover_point
                    )
                    if hover_point in index:
                        pass
                    else:
                        index.append(hover_point)
                index.sort()
            else:
                hover_point = list(self._indices_view).index(self._hover_point)
                index = [hover_point]

            # Color the hovered or selected points
            data = self._data_view[index]
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            size = self._sizes_view[index]
            face_color = [
                self.face_colors[i] for i in self._indices_view[index]
            ]
        else:
            data = np.empty((0, 2))
            size = 1
            face_color = 'white'

        width = self._highlight_width

        self._node._subvisuals[1].set_data(
            data[:, [1, 0]],
            size=size,
            edge_width=width,
            symbol=self.symbol,
            edge_color=self._highlight_color,
            face_color=face_color,
            scaling=True,
        )

        pos = self._selected_box
        if pos is None and not self._is_selecting:
            width = 0
            pos = np.empty((4, 2))
        elif self._is_selecting:
            pos = create_box(self._drag_box)

        pos = pos[list(range(4)) + [0]]

        self._node._subvisuals[0].set_data(
            pos=pos[:, [1, 0]], color=self._highlight_color, width=width
        )

    def get_message(self, coord, value):
        """Return coordinate and value string.

        Parameters
        ----------
        coord : sequence of int
            Position of mouse cursor in data.
        value : int
            Index of the point at the coord if any.

        Returns
        ----------
        msg : string
            String containing a message that can be used as
            a status update.
        """
        int_coord = np.round(coord).astype(int)
        msg = f'{int_coord}, {self.name}'
        if value is None:
            pass
        else:
            msg = msg + ', index ' + str(value)
        return msg

    def _update_thumbnail(self):
        """Update thumbnail with current points and colors."""
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        if len(self._data_view) > 0:
            min_vals = [self.range[-2][0], self.range[-1][0]]
            shape = np.ceil(
                [
                    self.range[-2][1] - self.range[-2][0] + 1,
                    self.range[-1][1] - self.range[-1][0] + 1,
                ]
            ).astype(int)
            zoom_factor = np.divide(self._thumbnail_shape[:2], shape).min()
            coords = np.floor(
                (self._data_view - min_vals + 0.5) * zoom_factor
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
            if self._hover_point in self.selected_data:
                self._hover_point = None
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
            if self._drag_start is None:
                center = self.data[index, -2:].mean(axis=0)
                self._drag_start = np.array(coord[-2:]) - center
            center = self.data[index, -2:].mean(axis=0)
            shift = coord[-2:] - center - self._drag_start
            self.data[index, -2:] = self.data[index, -2:] + shift
            self.refresh()

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
                'indices': self.indices,
            }
        else:
            self._clipboard = {}

    def _paste_data(self):
        """Paste any point from clipboard and select them."""
        npoints = len(self._data_view)
        totpoints = len(self.data)

        if len(self._clipboard.keys()) > 0:
            data = deepcopy(self._clipboard['data'])
            offset = np.subtract(
                self.indices[:-2], self._clipboard['indices'][:-2]
            )
            data[:, :-2] = data[:, :-2] + offset
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
            self.refresh()

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
            cx = str(d[1])
            cy = str(d[0])
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
        if event.pos is None:
            return
        self.position = tuple(event.pos)

        if self._mode == Mode.SELECT:
            if event.is_dragging:
                if len(self.selected_data) > 0:
                    self._move(self.selected_data, self.coordinates)
                else:
                    self._is_selecting = True
                    if self._drag_start is None:
                        self._drag_start = self.coordinates[-2:]
                    self._drag_box = np.array(
                        [self._drag_start, self.coordinates[-2:]]
                    )
                    self._set_highlight()
            else:
                self._hover_point = self.get_value()
                self._set_highlight()
        else:
            self._hover_point = self.get_value()
        self.status = self.get_message(self.coordinates, self._hover_point)

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.
        """
        if event.pos is None:
            return
        self.position = tuple(event.pos)
        shift = 'Shift' in event.modifiers

        if self._mode == Mode.SELECT:
            point = self.get_value()
            if shift and point is not None:
                if point in self.selected_data:
                    self.selected_data -= [point]
                else:
                    self.selected_data += [point]
            elif point is not None:
                if point not in self.selected_data:
                    self.selected_data = [point]
            else:
                self.selected_data = []
            self._set_highlight()
        elif self._mode == Mode.ADD:
            self.add(self.coordinates)

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.
        """
        if event.pos is None:
            return
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

    def on_key_press(self, event):
        """Called whenever key pressed in canvas.
        """
        if event.native.isAutoRepeat():
            return
        else:
            if event.key == ' ':
                if self._mode != Mode.PAN_ZOOM:
                    self._mode_history = self.mode
                    self._selected_data_history = copy(self.selected_data)
                    self.mode = Mode.PAN_ZOOM
                else:
                    self._mode_history = Mode.PAN_ZOOM
            elif event.key == 'Shift':
                if self._mode == Mode.ADD:
                    self.cursor = 'forbidden'
            elif event.key == 'p':
                self.mode = Mode.ADD
            elif event.key == 's':
                self.mode = Mode.SELECT
            elif event.key == 'z':
                self.mode = Mode.PAN_ZOOM
            elif event.key == 'c' and 'Control' in event.modifiers:
                if self._mode == Mode.SELECT:
                    self._copy_data()
            elif event.key == 'v' and 'Control' in event.modifiers:
                if self._mode == Mode.SELECT:
                    self._paste_data()
            elif event.key == 'a':
                if self._mode == Mode.SELECT:
                    self.selected_data = self._indices_view[
                        : len(self._data_view)
                    ]
                    self._set_highlight()
            elif event.key == 'Backspace':
                if self._mode == Mode.SELECT:
                    self.remove_selected()

    def on_key_release(self, event):
        """Called whenever key released in canvas.
        """
        if event.key == ' ':
            if self._mode_history != Mode.PAN_ZOOM:
                self.mode = self._mode_history
                self.selected_data = self._selected_data_history
                self._set_highlight()


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
            points + np.array([sizes / 2, sizes / 2]).T,
            points + np.array([sizes / 2, -sizes / 2]).T,
            points + np.array([-sizes / 2, sizes / 2]).T,
            points + np.array([-sizes / 2, -sizes / 2]).T,
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
