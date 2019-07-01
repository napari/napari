from typing import Union
from xml.etree.ElementTree import Element
import numpy as np
import itertools
from copy import copy, deepcopy
from contextlib import contextmanager
from scipy import ndimage as ndi
from .._base_layer import Layer
from ..._vispy.scene.visuals import Line, Markers, Compound
from ...util.event import Event
from ...util.misc import ensure_iterable
from vispy.color import get_color_names, Color
from ._constants import Symbol, SYMBOL_ALIAS, Mode


class Points(Layer):
    """Points layer.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates for each point.
    symbol : Symbol or {'arrow', 'clobber', 'cross', 'diamond', 'disc',
                         'hbar', 'ring', 'square', 'star', 'tailed_arrow',
                         'triangle_down', 'triangle_up', 'vbar', 'x'}
        Symbol to be used as a point. If given as a string, must be one of
        the following: arrow, clobber, cross, diamond, disc, hbar, ring,
        square, star, tailed_arrow, triangle_down, triangle_up, vbar, x
    size : int, float, np.ndarray, list
        Size of the point marker. If given as a scalar, all points are the same
        size. If given as a list/array, size must be the same length as
        coords and sets the point marker size for each point in coords
        (element-wise). If n_dimensional is True then can be a list of
        length dims or can be an array of shape Nxdims where N is the
        number of points and dims is the number of dimensions
    edge_width : int, float, None
        Width of the symbol edge in pixels.
    edge_color : Color, ColorArray
        Color of the point marker border.
    face_color : Color, ColorArray
        Color of the point marker body.
    n_dimensional : bool
        If True, renders points not just in central plane but also in all
        n-dimensions according to specified point marker size.

    Notes
    -----
    See vispy's marker visual docs for more details:
    http://api.vispy.org/en/latest/visuals.html#vispy.visuals.MarkersVisual
    """

    _highlight_color = (0, 0.6, 1)
    _highlight_width = 1.5

    def __init__(
        self,
        coords,
        symbol='o',
        size=10,
        edge_width=1,
        edge_color='black',
        face_color='white',
        n_dimensional=False,
        *,
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
        self._update_properties = True

        # Freeze refreshes
        with self.freeze_refresh():
            # Save the point coordinates
            self._coords = coords

            # Save the point style params
            self.symbol = symbol
            self.n_dimensional = n_dimensional
            self.edge_width = edge_width

            self.size_array = size
            self._edge_color_list = list(
                itertools.islice(
                    ensure_iterable(edge_color, color=True), 0, len(coords)
                )
            )
            self._face_color_list = list(
                itertools.islice(
                    ensure_iterable(face_color, color=True), 0, len(coords)
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

            # Indices of selected points within the currently viewed slice
            self._selected_points = []
            self._selected_points_stored = []
            self._selected_points_history = []
            self._cur_selected = []
            # Index of hovered point within the currently viewed slice
            self._hover_point = None
            self._hover_point_stored = None
            self._selected_box = None
            self._mode = Mode.PAN_ZOOM
            self._mode_history = self._mode
            self._status = self.mode

            self._drag_start = None

            # Nx2 array of points in the currently viewed slice
            self._points_view = np.empty((0, 2))
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

        self.events.opacity.connect(lambda e: self._update_thumbnail())

    @property
    def coords(self) -> np.ndarray:
        """ndarray: coordinates of the point centroids
        """
        return self._coords

    @coords.setter
    def coords(self, coords: np.ndarray):
        cur_npoints = len(self._coords)
        self._coords = coords

        # Adjust the size array when the number of points has changed
        if len(coords) < cur_npoints:
            # If there are now less points, remove the sizes and colors of the
            # extra ones
            with self.freeze_refresh():
                self.size_array = self._size_array[: len(coords)]
                self._edge_color_list = self._edge_color_list[: len(coords)]
                self._face_color_list = self._face_color_list[: len(coords)]

        elif len(coords) > cur_npoints:
            # If there are now more points, add the sizes and colors of the
            # new ones
            with self.freeze_refresh():
                adding = len(coords) - cur_npoints
                if len(self._size_array) > 0:
                    new_size = copy(self._size_array[-1])
                    new_size[-2:] = self.size
                else:
                    # Add the default size, with a value for each dimension
                    new_size = np.repeat(self.size, self._size_array.shape[1])
                size = np.repeat([new_size], adding, axis=0)
                self.size_array = np.concatenate(
                    (self._size_array, size), axis=0
                )
                self._edge_color_list += [
                    self.edge_color for i in range(adding)
                ]
                self._face_color_list += [
                    self.face_color for i in range(adding)
                ]

        self.events.data()
        self.refresh()

    @property
    def data(self) -> np.ndarray:
        """ndarray: coordinates of the point centroids
        """
        return self._coords

    @data.setter
    def data(self, data: np.ndarray) -> None:
        self.coords = data

    @property
    def n_dimensional(self) -> str:
        """ bool: if True, renders points not just in central plane but also
        in all n dimensions according to specified point marker size
        """
        return self._n_dimensional

    @n_dimensional.setter
    def n_dimensional(self, n_dimensional: bool) -> None:
        self._n_dimensional = n_dimensional
        self.events.n_dimensional()

        self.refresh()

    @property
    def symbol(self) -> str:
        """ str: point symbol
        """
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

    @contextmanager
    def block_update_properties(self):
        self._update_properties = False
        yield
        self._update_properties = True

    @property
    def size_array(self) -> Union[int, float, np.ndarray, list]:
        """ndarray: size of the point marker symbol in px
        """
        return self._size_array

    @size_array.setter
    def size_array(self, size: Union[int, float, np.ndarray, list]) -> None:
        try:
            self._size_array = np.broadcast_to(size, self._coords.shape).copy()
        except:
            try:
                self._size_array = np.broadcast_to(
                    size, self._coords.shape[::-1]
                ).T.copy()
            except:
                raise ValueError("Size is not compatible for broadcasting")
        self.refresh()

    @property
    def size(self) -> Union[int, float]:
        """int, float: size of the point in px
        """
        return self._size

    @size.setter
    def size(self, size: Union[None, float]) -> None:
        self._size = size
        if self._update_properties and len(self.selected_points) > 0:
            index = self._indices_view[self.selected_points]
            for i in index:
                self.size_array[i, :] = (self.size_array[i, :] > 0) * size
            self.refresh()
        self.events.size()

    @property
    def edge_width(self) -> Union[None, int, float]:
        """None, int, float, None: width of the symbol edge in px
        """

        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: Union[None, float]) -> None:
        self._edge_width = edge_width

        self.refresh()

    @property
    def edge_color(self) -> str:
        """Color, ColorArray: the point marker edge color
        """

        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str) -> None:
        self._edge_color = edge_color
        if self._update_properties and len(self.selected_points) > 0:
            index = self._indices_view[self.selected_points]
            for i in index:
                self._edge_color_list[i] = edge_color
            self.refresh()
        self.events.edge_color()

    @property
    def face_color(self) -> str:
        """Color, ColorArray: color of the body of the point marker body
        """

        return self._face_color

    @face_color.setter
    def face_color(self, face_color: str) -> None:
        self._face_color = face_color
        if self._update_properties and len(self.selected_points) > 0:
            index = self._indices_view[self.selected_points]
            for i in index:
                self._face_color_list[i] = face_color
            self.refresh()
        self.events.face_color()

    @property
    def selected_points(self):
        """list: list of currently selected points
        """
        return self._selected_points

    @selected_points.setter
    def selected_points(self, selected_points):
        self._selected_points = selected_points
        self._selected_box = self.interaction_box(selected_points)
        if len(self._indices_view) > 0:
            index = self._indices_view[self._selected_points]
            self._cur_selected = self._indices_view[self._selected_points]
        else:
            index = []
            self._cur_selected = []

        # Update properties based on selected points
        edge_colors = list(set([self._edge_color_list[i] for i in index]))
        if len(edge_colors) == 1:
            edge_color = edge_colors[0]
            with self.block_update_properties():
                self.edge_color = edge_color

        face_colors = list(set([self._face_color_list[i] for i in index]))
        if len(face_colors) == 1:
            face_color = face_colors[0]
            with self.block_update_properties():
                self.face_color = face_color

        size = list(set([self.size_array[i, -2:].mean() for i in index]))
        if len(size) == 1:
            size = size[0]
            with self.block_update_properties():
                self.size = size

    def interaction_box(self, index):
        """Create the interaction box around a list of points.

        Parameters
        ----------
        index : list
            List of points around which to construct the interaction box.

        Returns
        ----------
        box : np.ndarray
            5x2 array of corners of the interaction box in clockwise order
            starting in the upper-left corner and duplicating the first corner
        """
        if len(index) == 0:
            box = None
        else:
            data = self._points_view[index]
            size = self._sizes_view[index]
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            data = points_to_squares(data, size)
            min_val = np.array(
                [data[:, 0].min(axis=0), data[:, 1].min(axis=0)]
            )
            max_val = np.array(
                [data[:, 0].max(axis=0), data[:, 1].max(axis=0)]
            )

            tl = np.array([min_val[0], min_val[1]])
            tr = np.array([max_val[0], min_val[1]])
            br = np.array([max_val[0], max_val[1]])
            bl = np.array([min_val[0], max_val[1]])

            box = np.array([tl, tr, br, bl, tl])

        return box

    @property
    def svg_props(self):
        """dict: opacity and width properties in the svg specification
        """
        width = str(self.edge_width)
        opacity = str(self.opacity)

        # Currently not using fill or stroke opacity - only global opacity
        # as otherwise leads to unexpected behavior when reading svg into
        # other applications
        # fill_opacity = f'{self.opacity*self.face_color.rgba[3]}'
        # stroke_opacity = f'{self.opacity*self.edge_color.rgba[3]}'

        props = {'stroke-width': width, 'opacity': opacity}

        return props

    @property
    def mode(self):
        """None, str: Interactive mode
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
            self.selected_points = []
            self._set_highlight()

        self.status = str(mode)
        self._mode = mode

        self.events.mode(mode=mode)

    def _get_shape(self):
        if len(self.coords) == 0:
            # when no points given, return (1,) * dim
            # we use coords.shape[1] as the dimensionality of the image
            return np.ones(self.coords.shape[1], dtype=int)
        else:
            return np.max(self.coords, axis=0) + 1

    @property
    def range(self):
        """list of 3-tuple of int: ranges of data for slicing specifed by
        (min, max, step).
        """
        if len(self.coords) == 0:
            maxs = np.ones(self.coords.shape[1], dtype=int)
            mins = np.zeros(self.coords.shape[1], dtype=int)
        else:
            maxs = np.max(self.coords, axis=0) + 1
            mins = np.min(self.coords, axis=0)

        return [(min, max, 1) for min, max in zip(mins, maxs)]

    def _slice_points(self, indices):
        """Determines the slice of points given the indices.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.

        Returns
        ----------
        in_slice_points : (N, 2) array
            Coordinates of points in the currently viewed slice.
        slice_indices : list
            Indices of points in the currently viewed slice.
        scale : float, (N, ) array
            If in `n_dimensional` mode then the scale factor of points, where
            values of 1 corresponds to points located in the slice, and values
            less than 1 correspond to points located in neighboring slices.
        """
        # Get a list of the coords for the points in this slice
        coords = self.data
        if len(coords) > 0:
            if self.n_dimensional is True and self.ndim > 2:
                distances = abs(coords[:, :-2] - indices[:-2])
                size_array = self.size_array[:, :-2] / 2
                matches = np.all(distances <= size_array, axis=1)
                in_slice_points = coords[matches, -2:]
                size_match = size_array[matches]
                size_match[size_match == 0] = 1
                scale_per_dim = (size_match - distances[matches]) / size_match
                scale_per_dim[size_match == 0] = 1
                scale = np.prod(scale_per_dim, axis=1)
                indices = np.where(matches)[0].astype(int)
                return in_slice_points, indices, scale
            else:
                matches = np.all(coords[:, :-2] == indices[:-2], axis=1)
                in_slice_points = coords[matches, -2:]
                indices = np.where(matches)[0].astype(int)
                return in_slice_points, indices, 1
        else:
            return [], [], []

    def _select_point(self, coord):
        """Determines selected points selected given indices.

        Parameters
        ----------
        coord : 2-tuple
            Indices to check if point at for currently viewed points.

        Returns
        ----------
        selection : int or None
            Index of point that is at the current coordinate if any.
        """
        in_slice_points = self._points_view

        # Display points if there are any in this slice
        if len(self._points_view) > 0:
            # Get the point sizes
            distances = abs(self._points_view - coord)
            in_slice_matches = np.all(
                distances <= np.expand_dims(self._sizes_view, axis=1) / 2,
                axis=1,
            )
            indices = np.where(in_slice_matches)[0]
            if len(indices) > 0:
                selection = indices[-1]
            else:
                selection = None
        else:
            selection = None

        return selection

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""

        in_slice_points, indices, scale = self._slice_points(self.indices)

        # Display points if there are any in this slice
        if len(in_slice_points) > 0:
            # Get the point sizes
            sizes = self.size_array[indices, -2:].mean(axis=1) * scale

            # Update the points node
            data = np.array(in_slice_points) + 0.5

        else:
            # if no points in this slice send dummy data
            data = np.empty((0, 2))
            sizes = [0]
        self._points_view = data
        self._sizes_view = sizes
        self._indices_view = indices
        # Make sure if changing planes any selected points not in the current
        # plane are removed
        selected = []
        for c in self._cur_selected:
            if c in self._indices_view:
                ind = list(self._indices_view).index(c)
                selected.append(ind)
        self._selected_points = selected
        self._selected_box = self.interaction_box(self.selected_points)
        if len(self.selected_points) > 0:
            self._cur_selected = self._indices_view[self.selected_points]
        else:
            self._cur_selected = []

        if len(data) > 0:
            edge_color = [
                self._edge_color_list[i] for i in self._indices_view[::-1]
            ]
            face_color = [
                self._face_color_list[i] for i in self._indices_view[::-1]
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
            self.selected_points == self._selected_points_stored
            and self._hover_point == self._hover_point_stored
            and np.all(self._drag_box == self._drag_box_stored)
        ) and not force:
            return
        self._selected_points_stored = copy(self.selected_points)
        self._hover_point_stored = copy(self._hover_point)
        self._drag_box_stored = copy(self._drag_box)

        if self._mode == Mode.SELECT and (
            self._hover_point is not None or len(self.selected_points) > 0
        ):
            if len(self.selected_points) > 0:
                index = copy(self.selected_points)
                if self._hover_point is not None:
                    if self._hover_point in index:
                        pass
                    else:
                        index.append(self._hover_point)
                index.sort()
            else:
                index = [self._hover_point]

            # Color the hovered or selected points
            data = self._points_view[index]
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            size = self._sizes_view[index]
            face_color = [
                self._face_color_list[i] for i in self._indices_view[index]
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
            pos = np.empty((0, 2))
        elif self._is_selecting:
            pos = create_box(self._drag_box)[list(range(4)) + [0]]

        self._node._subvisuals[0].set_data(
            pos=pos[:, [1, 0]], color=self._highlight_color, width=width
        )

    def get_message(self, coord, value):
        """Returns coordinate and value string for given mouse coordinates
        and value.

        Parameters
        ----------
        coord : sequence of int
            Position of mouse cursor in data.
        value : int or float or sequence of int or float
            Value of the data at the coord.

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
            # map from the index of the point in the slice to the index of the
            # point in the full list of points
            index = self._indices_view[value]
            msg = msg + ', index ' + str(index)
        return msg

    def _update_thumbnail(self):
        """Update thumbnail with current points and colors.
        """
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        if len(self._points_view) > 0:
            min_vals = [self.range[-2][0], self.range[-1][0]]
            shape = np.ceil(
                [
                    self.range[-2][1] - self.range[-2][0] + 1,
                    self.range[-1][1] - self.range[-1][0] + 1,
                ]
            ).astype(int)
            zoom_factor = np.divide(self._thumbnail_shape[:2], shape).min()
            coords = np.floor(
                (self._points_view - min_vals + 0.5) * zoom_factor
            ).astype(int)
            coords = np.clip(
                coords, 0, np.subtract(self._thumbnail_shape[:2], 1)
            )
            for i, c in enumerate(coords):
                col = self._face_color_list[self._indices_view[i]]
                colormapped[c[0], c[1], :] = Color(col).rgba
        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    def _add(self, coord):
        """Adds point at given mouse position
        and set of indices.
        Parameters
        ----------
        coord : sequence of indices to add point at
        """
        self.data = np.append(self.data, [coord], axis=0)

    def remove_selected(self):
        """Removes selected points if any.
        """
        index = self._indices_view[self.selected_points]
        if index is not None:
            self._size_array = np.delete(self._size_array, index, axis=0)
            for i in index[::-1]:
                del self._edge_color_list[i]
                del self._face_color_list[i]
            if self._hover_point in self.selected_points:
                self._hover_point = None
            self.selected_points = []
            self.data = np.delete(self.data, index, axis=0)

    def _move(self, coord):
        """Moves selected points.

        Parameters
        ----------
        coord : sequence of indices to move points to
        """
        index = self._indices_view[self.selected_points]
        if len(index) > 0:
            if self._drag_start is None:
                center = self.data[index, -2:].mean(axis=0)
                self._drag_start = np.array(coord[-2:]) - center
            center = self.data[index, -2:].mean(axis=0)
            shift = coord[-2:] - center - self._drag_start
            self.data[index, -2:] = self.data[index, -2:] + shift
            self.refresh()
        else:
            self._is_selecting = True
            if self._drag_start is None:
                self._drag_start = coord[-2:]
            self._drag_box = np.array([self._drag_start, coord[-2:]])
            self._set_highlight()

    def _copy_points(self):
        """Copy selected points to clipboard.
        """
        if len(self.selected_points) > 0:
            index = self._indices_view[self.selected_points]
            self._clipboard = {
                'coord': deepcopy(self.data[index]),
                'size': deepcopy(self.size_array[index]),
                'edge_color': deepcopy(
                    [self._edge_color_list[i] for i in index]
                ),
                'face_color': deepcopy(
                    [self._face_color_list[i] for i in index]
                ),
                'indices': self.indices,
            }
        else:
            self._clipboard = {}

    def _paste_points(self):
        """Paste any point from clipboard and select them.
        """
        npoints = len(self._points_view)
        totpoints = len(self.data)

        if len(self._clipboard.keys()) > 0:
            coords = deepcopy(self._clipboard['coord'])
            offset = np.subtract(
                self.indices[:-2], self._clipboard['indices'][:-2]
            )
            coords[:, :-2] = coords[:, :-2] + offset
            self._coords = np.append(self.data, coords, axis=0)
            self._size_array = np.append(
                self.size_array, deepcopy(self._clipboard['size']), axis=0
            )
            self._edge_color_list = self._edge_color_list + deepcopy(
                self._clipboard['edge_color']
            )
            self._face_color_list = self._face_color_list + deepcopy(
                self._clipboard['face_color']
            )
            self._selected_points = list(
                range(npoints, npoints + len(self._clipboard['coord']))
            )
            self._cur_selected = list(
                range(totpoints, totpoints + len(self._clipboard['coord']))
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

        for i, d, s in zip(
            self._indices_view, self._points_view, self._sizes_view
        ):
            cx = str(d[1])
            cy = str(d[0])
            r = str(s / 2)
            face_color = (255 * Color(self._face_color_list[i]).rgba).astype(
                np.int
            )
            fill = f'rgb{tuple(face_color[:3])}'
            edge_color = (255 * Color(self._edge_color_list[i]).rgba).astype(
                np.int
            )
            stroke = f'rgb{tuple(edge_color[:3])}'
            element = Element(
                'circle',
                cx=cx,
                cy=cy,
                r=r,
                stroke=stroke,
                fill=fill,
                **self.svg_props,
            )
            xml_list.append(element)

        return xml_list

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        if event.pos is None:
            return
        self.position = tuple(event.pos)
        coord = self.coordinates

        if self._mode == Mode.SELECT:
            if event.is_dragging:
                self._move(coord)
            else:
                self._hover_point = self._select_point(coord[-2:])
                self._set_highlight()
        else:
            self._hover_point = self._select_point(coord[-2:])
        self.status = self.get_message(coord, self._hover_point)

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.
        """
        if event.pos is None:
            return
        self.position = tuple(event.pos)
        coord = self.coordinates
        shift = 'Shift' in event.modifiers

        if self._mode == Mode.SELECT:
            point = self._select_point(coord[-2:])
            if shift and point is not None:
                if point in self.selected_points:
                    self.selected_points -= [point]
                else:
                    self.selected_points += [point]
            elif point is not None:
                if point not in self.selected_points:
                    self.selected_points = [point]
            else:
                self.selected_points = []
            self._set_highlight()
        elif self._mode == Mode.ADD:
            self._add(coord)

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.
        """
        if event.pos is None:
            return
        self._drag_start = None
        if self._is_selecting:
            self._is_selecting = False
            self.selected_points = points_in_box(
                self._drag_box, self._points_view, self._sizes_view
            )
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
                    self._selected_points_history = copy(self.selected_points)
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
                    self._copy_points()
            elif event.key == 'v' and 'Control' in event.modifiers:
                if self._mode == Mode.SELECT:
                    self._paste_points()
            elif event.key == 'a':
                if self._mode == Mode.SELECT:
                    self.selected_points = list(range(len(self._points_view)))
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
                self.selected_points = self._selected_points_history
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
