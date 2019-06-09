from typing import Union
from xml.etree.ElementTree import Element
import numpy as np
from .._base_layer import Layer
from ..._vispy.scene.visuals import Markers as MarkersNode
from ...util.event import Event
from vispy.color import get_color_names, Color
from ._constants import Symbol, SYMBOL_ALIAS, Mode


class Markers(Layer):
    """Markers layer.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates for each marker.
    symbol : Symbol or {'arrow', 'clobber', 'cross', 'diamond', 'disc',
                         'hbar', 'ring', 'square', 'star', 'tailed_arrow',
                         'triangle_down', 'triangle_up', 'vbar', 'x'}
        Symbol to be used as a marker. If given as a string, must be one of
        the following: arrow, clobber, cross, diamond, disc, hbar, ring,
        square, star, tailed_arrow, triangle_down, triangle_up, vbar, x
    size : int, float, np.ndarray, list
        Size of the marker. If given as a scalar, all markers are the same
        size. If given as a list/array, size must be the same length as
        coords and sets the marker size for each marker in coords
        (element-wise). If n_dimensional is True then can be a list of
        length dims or can be an array of shape Nxdims where N is the
        number of markers and dims is the number of dimensions
    edge_width : int, float, None
        Width of the symbol edge in pixels.
    edge_color : Color, ColorArray
        Color of the marker border.
    face_color : Color, ColorArray
        Color of the marker body.
    n_dimensional : bool
        If True, renders markers not just in central plane but also in all
        n-dimensions according to specified marker size.

    Notes
    -----
    See vispy's marker visual docs for more details:
    http://api.vispy.org/en/latest/visuals.html#vispy.visuals.MarkersVisual
    """

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
        super().__init__(MarkersNode(), name)

        self.events.add(
            mode=Event,
            size=Event,
            face_color=Event,
            edge_color=Event,
            symbol=Event,
            n_dimensional=Event,
        )

        # Freeze refreshes
        with self.freeze_refresh():
            # Save the marker coordinates
            self._coords = coords

            # Save the marker style params
            self.symbol = symbol
            self.size = size
            self.edge_width = edge_width
            self.edge_color = edge_color
            self.face_color = face_color
            self.n_dimensional = n_dimensional
            self._colors = get_color_names()
            self._selected_markers = None
            self._mode = Mode.PAN_ZOOM
            self._mode_history = self._mode
            self._status = self._mode
            self._markers_view = np.empty((0, 2))
            self._sizes_view = 0

            # update flags
            self._need_display_update = False
            self._need_visual_update = False

    @property
    def coords(self) -> np.ndarray:
        """ndarray: coordinates of the marker centroids
        """
        return self._coords

    @coords.setter
    def coords(self, coords: np.ndarray):
        self._coords = coords

        # Adjust the size array when the number of markers has changed
        if len(coords) < len(self._size):
            # If there are now less markers, remove the sizes of the missing
            # ones
            with self.freeze_refresh():
                self.size = self._size[: len(coords)]
        elif len(coords) > len(self._size):
            # If there are now more markers, add the sizes of last one
            # or add the default size
            with self.freeze_refresh():
                adding = len(coords) - len(self._size)
                if len(self._size) > 0:
                    new_size = self._size[-1]
                else:
                    # Add the default size, with a value for each dimension
                    new_size = np.repeat(10, self._size.shape[1])
                size = np.repeat([new_size], adding, axis=0)
                self.size = np.concatenate((self._size, size), axis=0)
        self.events.data()
        self.refresh()

    @property
    def data(self) -> np.ndarray:
        """ndarray: coordinates of the marker centroids
        """
        return self._coords

    @data.setter
    def data(self, data: np.ndarray) -> None:
        self.coords = data

    @property
    def n_dimensional(self) -> str:
        """ bool: if True, renders markers not just in central plane but also
        in all n dimensions according to specified marker size
        """
        return self._n_dimensional

    @n_dimensional.setter
    def n_dimensional(self, n_dimensional: bool) -> None:
        self._n_dimensional = n_dimensional
        self.events.n_dimensional()

        self.refresh()

    @property
    def symbol(self) -> str:
        """ str: marker symbol
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

    @property
    def size(self) -> Union[int, float, np.ndarray, list]:
        """float, ndarray: size of the marker symbol in px
        """

        return self._size_original

    @size.setter
    def size(self, size: Union[int, float, np.ndarray, list]) -> None:

        try:
            self._size = np.broadcast_to(size, self._coords.shape)
        except:
            try:
                self._size = np.broadcast_to(size, self._coords.shape[::-1]).T
            except:
                raise ValueError("Size is not compatible for broadcasting")
        self._size_original = size
        self.events.size()

        self.refresh()

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
        """Color, ColorArray: the marker edge color
        """

        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str) -> None:
        self._edge_color = edge_color
        self.events.edge_color()

        self.refresh()

    @property
    def face_color(self) -> str:
        """Color, ColorArray: color of the body of the marker
        """

        return self._face_color

    @face_color.setter
    def face_color(self, face_color: str) -> None:
        self._face_color = face_color
        self.events.face_color()

        self.refresh()

    @property
    def svg_props(self):
        """dict: color and width properties in the svg specification
        """
        width = str(self.edge_width)
        face_color = (255 * Color(self.face_color).rgba).astype(np.int)
        fill = f'rgb{tuple(face_color[:3])}'
        edge_color = (255 * Color(self.edge_color).rgba).astype(np.int)
        stroke = f'rgb{tuple(edge_color[:3])}'
        opacity = str(self.opacity)

        # Currently not using fill or stroke opacity - only global opacity
        # as otherwise leads to unexpected behavior when reading svg into
        # other applications
        # fill_opacity = f'{self.opacity*self.face_color.rgba[3]}'
        # stroke_opacity = f'{self.opacity*self.edge_color.rgba[3]}'

        props = {
            'fill': fill,
            'stroke': stroke,
            'stroke-width': width,
            'opacity': opacity,
        }

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

        if mode == Mode.ADD:
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom'
        elif mode == Mode.SELECT:
            self.cursor = 'pointing'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom'
        elif mode == Mode.PAN_ZOOM:
            self.cursor = 'standard'
            self.interactive = True
            self.help = ''
        else:
            raise ValueError("Mode not recognized")

        self.status = str(mode)
        self._mode = mode

        self.events.mode(mode=mode)

    def _get_shape(self):
        if len(self.coords) == 0:
            # when no markers given, return (1,) * dim
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

    def _slice_markers(self, indices):
        """Determines the slice of markers given the indices.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.
        """
        # Get a list of the coords for the markers in this slice
        coords = self.coords
        if len(coords) > 0:
            if self.n_dimensional is True and self.ndim > 2:
                distances = abs(coords[:, :-2] - indices[:-2])
                size_array = self._size[:, :-2] / 2
                matches = np.all(distances <= size_array, axis=1)
                in_slice_markers = coords[matches, -2:]
                size_match = size_array[matches]
                size_match[size_match == 0] = 1
                scale_per_dim = (size_match - distances[matches]) / size_match
                scale_per_dim[size_match == 0] = 1
                scale = np.prod(scale_per_dim, axis=1)
                return in_slice_markers, matches, scale
            else:
                matches = np.all(coords[:, :-2] == indices[:-2], axis=1)
                in_slice_markers = coords[matches, -2:]
                return in_slice_markers, matches, 1
        else:
            return [], [], []

    def _select_marker(self, indices):
        """Determines selected markers selected given indices.

        Parameters
        ----------
        indices : sequence of int
            Indices to check if marker at.
        """
        in_slice_markers, matches, scale = self._slice_markers(indices)

        # Display markers if there are any in this slice
        if len(in_slice_markers) > 0:
            # Get the marker sizes
            size_array = self._size[matches, -2:] * np.expand_dims(
                scale, axis=1
            )
            distances = abs(in_slice_markers - indices[-2:])
            in_slice_matches = np.all(distances <= size_array / 2, axis=1)
            indices = np.where(in_slice_matches)[0]
            if len(indices) > 0:
                selection = np.where(matches)[0][indices[-1]]
            else:
                selection = None
        else:
            selection = None

        return selection

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""

        in_slice_markers, matches, scale = self._slice_markers(self.indices)

        # Display markers if there are any in this slice
        if len(in_slice_markers) > 0:
            # Get the marker sizes
            sizes = (self._size[matches, -2:].mean(axis=1) * scale)[::-1]

            # Update the markers node
            data = np.array(in_slice_markers)[::-1] + 0.5

        else:
            # if no markers in this slice send dummy data
            data = np.empty((0, 2))
            sizes = 0
        self._markers_view = data
        self._sizes_view = sizes

        self._node.set_data(
            data[:, [1, 0]],
            size=sizes,
            edge_width=self.edge_width,
            symbol=self.symbol,
            edge_color=self.edge_color,
            face_color=self.face_color,
            scaling=True,
        )
        self._need_visual_update = True
        self._update()
        self.status = self.get_message(
            self.coordinates, self._selected_markers
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
            msg = msg + ', index ' + str(value)
        return msg

    def _add(self, coord):
        """Adds object at given mouse position
        and set of indices.
        Parameters
        ----------
        coord : sequence of indices to add marker at
        """
        self.data = np.append(self.data, [coord], axis=0)
        self._selected_markers = len(self.data) - 1

    def _remove(self):
        """Removes selected object if any.
        """
        index = self._selected_markers
        if index is not None:
            self._size = np.delete(self._size, index, axis=0)
            self.data = np.delete(self.data, index, axis=0)
            self._selected_markers = None

    def _move(self, coord):
        """Moves object at given mouse position
        and set of indices.
        Parameters
        ----------
        coord : sequence of indices to move marker to
        """
        index = self._selected_markers
        if index is not None:
            self.data[index] = coord
            self.refresh()

    def to_xml_list(self):
        """Convert the markers to a list of xml elements according to the svg
        specification. Z ordering of the markers will be taken into account.
        Each marker is represented by a circle. Support for other symbols is
        not yet implemented.

        Returns
        ----------
        xml : list
            List of xml elements defining each marker according to the
            svg specification
        """
        xml_list = []

        for d, s in zip(self._markers_view, self._sizes_view):
            cx = str(d[1])
            cy = str(d[0])
            r = str(s / 2)
            element = Element('circle', cx=cx, cy=cy, r=r, **self.svg_props)
            xml_list.append(element)

        return xml_list

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        if event.pos is None:
            return
        self.position = tuple(event.pos)
        coord = self.coordinates
        if self._mode == Mode.SELECT and event.is_dragging:
            self._move(coord)
        else:
            self._selected_markers = self._select_marker(coord)
        self.status = self.get_message(coord, self._selected_markers)

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.
        """
        if event.pos is None:
            return
        self.position = tuple(event.pos)
        coord = self.coordinates
        self._selected_markers = self._select_marker(coord)
        shift = 'Shift' in event.modifiers

        if self._mode == Mode.ADD:
            if shift:
                self._remove()
            else:
                self._add(coord)
        self.status = self.get_message(coord, self._selected_markers)

    def on_key_press(self, event):
        """Called whenever key pressed in canvas.
        """
        if event.native.isAutoRepeat():
            return
        else:
            if event.key == ' ':
                if self._mode != Mode.PAN_ZOOM:
                    self._mode_history = self.mode
                    self.mode = Mode.PAN_ZOOM
                else:
                    self._mode_history = Mode.PAN_ZOOM
            elif event.key == 'Shift':
                if self._mode == Mode.ADD:
                    self.cursor = 'forbidden'
            elif event.key == 'a':
                self.mode = Mode.ADD
            elif event.key == 's':
                self.mode = Mode.SELECT
            elif event.key == 'z':
                self.mode = Mode.PAN_ZOOM

    def on_key_release(self, event):
        """Called whenever key released in canvas.
        """
        if event.key == ' ':
            if self._mode_history != Mode.PAN_ZOOM:
                self.mode = self._mode_history
        elif event.key == 'Shift':
            if self.mode == 'add':
                self.cursor = 'cross'
