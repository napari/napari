from typing import Union, List, Tuple
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
from ._constants import Mode


class Text(Layer):
    """Annotations layer.

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
        annotations=None,
        annotation_offset=None,
        text_color='black',
        font_size=12,
        font='OpenSans',
        anchor_x='center',
        anchor_y='center',
        render_method='cpu',
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        opacity=1,
        blending='translucent',
        visible=True,
    ):
        if data is None:
            data = (np.empty((0, 2)), [])
        ndim = data[0].shape[1]
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
            mode=Event, text_color=Event, font_size=Event, highlight=Event
        )
        self._colors = get_color_names()

        # Save the text data
        self._data = data
        self.dims.clip = False
        self._coords = data[0]
        self._text = data[1]

        self._sizes = []

        # Save the text style params
        self.text_color = text_color
        self._font_size = font_size
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.sizes = font_size

        self.new_text = ''

        self.render_method = render_method

        # The following point properties are for the new points that will
        # be added. For any given property, if a list is passed to the
        # constructor so each point gets its own value then the default
        # value is used when adding new points

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

        # Full data indices of points located in the currently viewed slice
        self._indices_view = []

        self._drag_box = None
        self._drag_box_stored = None
        self._is_selecting = False
        self._clipboard = {}

        # Trigger generation of view slice and thumbnail
        self._update_dims()

    @property
    def data(self) -> np.ndarray:
        """(N, D) array: coordinates for N points in D dimensions."""
        return self._data

    @data.setter
    def data(self, data: Tuple[np.ndarray, List[str]]):
        cur_npoints = len(self._data)
        self._data = data
        self._coords = data[0]
        self._text = data[1]

        # Adjust the size array when the number of points has changed
        if len(data) < cur_npoints:
            # If there are now less points, remove the sizes and colors of the
            # extra ones
            # with self.events.set_data.blocker():
            #     self.edge_colors = self.edge_colors[: len(data)]
            #     self.face_colors = self.face_colors[: len(data)]
            #     self.sizes = self._sizes[: len(data)]
            pass

        elif len(data) > cur_npoints:
            # If there are now more points, add the sizes and colors of the
            # new ones
            with self.events.set_data.blocker():
                adding = len(data) - cur_npoints
                if len(self._sizes) > 0:
                    pass
                else:
                    # Add the default size, with a value for each dimension
                    pass
        self._update_sizes(self.font_size)
        self._update_dims()
        self.events.data()

    @property
    def coords(self) -> np.ndarray:
        return self._coords

    @property
    def text(self) -> List[str]:
        return self._text

    def _get_ndim(self):
        """Determine number of dimensions of the layer."""
        return self.data[0].shape[1]

    def _get_extent(self):
        """Determine ranges for slicing given by (min, max, step)."""
        if len(self.data) == 0:
            maxs = np.ones(self.data[0].shape[1], dtype=int)
            mins = np.zeros(self.data[0].shape[1], dtype=int)
        else:
            maxs = np.max(self.data[0], axis=0)
            mins = np.min(self.data[0], axis=0)

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
    def font_size(self):
        return self._font_size

    @font_size.setter
    def font_size(self, size: float):
        self._font_size = size
        self.sizes = size
        self._set_view_slice()
        self.events.font_size()
        self.events.highlight()

    @property
    def text_color(self) -> str:
        """str: face color of marker for the next added point."""

        return self._text_color

    @text_color.setter
    def text_color(self, text_color: str) -> None:
        self._text_color = text_color
        self.events.text_color()
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
            data = points_to_squares(data, size, self.scale_factor)
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

    @property
    def sizes(self):
        return self._sizes

    @sizes.setter
    def sizes(self, font_size):
        self._update_sizes(font_size)

    def _update_sizes(self, font_size):
        sizes = [
            (font_size * 2.2, font_size * len(t) * 1.1) for t in self.text
        ]

        self._sizes = sizes

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
        in_slice_coords : (N, 2) array
            Coordinates of text in the currently viewed slice.
        in_slice_text : (N) List[str]
            List of the text to display (index-matched to coords)
        slice_indices : list
            Indices of text in the currently viewed slice.
        """
        # Get a list of the data for the points in this slice
        not_disp = list(self.dims.not_displayed)
        disp = list(self.dims.displayed)
        indices = np.array(indices)
        if len(self.data) > 0:
            data = self.coords[:, not_disp].astype('int')
            matches = np.all(data == indices[not_disp], axis=1)
            in_slice_coords = self.coords[np.ix_(matches, disp)]
            slice_indices = np.where(matches)[0].astype(int)
            in_slice_text = [self.text[i] for i in slice_indices]
            return in_slice_coords, in_slice_text, slice_indices
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
        in_slice_sizes = self._sizes_view

        # Display points if there are any in this slice
        if len(self._data_view) > 0:
            # Get the point sizes
            distances = abs(
                self._data_view
                - [self.coordinates[d] for d in self.dims.displayed]
            )

            hitbox_half_width = in_slice_sizes / 2 * self.scale_factor

            in_slice_matches = np.all(distances <= hitbox_half_width, axis=1)
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

        in_slice_coords, in_slice_text, indices = self._slice_data(
            self.dims.indices
        )

        # Display points if there are any in this slice
        if len(in_slice_coords) > 0:
            # Update the points node
            data = np.array(in_slice_coords)

        else:
            # if no points in this slice send dummy data
            data = np.zeros((0, self.dims.ndisplay))

        self._data_view = data
        self._text_view = in_slice_text
        self._sizes_view = np.asarray([self.sizes[i] for i in indices])
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

        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    def add(self, coord):
        """Adds point at coordinate.

        Parameters
        ----------
        coord : sequence of indices to add point at
        """
        coords = np.append(self.coords, [coord], axis=0)
        text = self.text

        new_text = self.new_text
        if self.new_text == '':
            new_text = 'EditMe'

        text.append(new_text)

        self.data = (coords, text)

    def remove_selected(self):
        """Removes selected points if any."""
        index = copy(self.selected_data)
        index.sort()
        if len(index) > 0:
            if self._value in self.selected_data:
                self._value = None
            self.selected_data = []
            coords = self.data[0]
            text = self.data[1]

            coords = np.delete(coords, index, axis=0)
            text = [t for i, t in enumerate(text) if i not in index]
            self.data = (coords, text)

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
                center = self.coords[np.ix_(index, disp)].mean(axis=0)
                self._drag_start = np.array(coord)[disp] - center
            center = self.coords[np.ix_(index, disp)].mean(axis=0)
            shift = np.array(coord)[disp] - center - self._drag_start
            self.data[0][np.ix_(index, disp)] = (
                self.coords[np.ix_(index, disp)] + shift
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
                    self._drag_box,
                    self._data_view,
                    self._sizes_view,
                    self.scale_factor,
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


def points_to_squares(points, sizes, scale_factor):
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
    hitbox_half_width = sizes / 2 * scale_factor

    rect = np.concatenate(
        [
            points + hitbox_half_width,
            points + np.multiply(hitbox_half_width, [1, -1]),
            points + np.multiply(hitbox_half_width, [-1, 1]),
            points - hitbox_half_width,
        ],
        axis=0,
    )

    return rect


def points_in_box(corners, points, sizes, scale_factor):
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
    rect = points_to_squares(points, sizes, scale_factor)
    below_top = np.all(box[1] >= rect, axis=1)
    above_bottom = np.all(rect >= box[0], axis=1)
    inside = np.logical_and(below_top, above_bottom)
    inside = np.unique(np.where(inside)[0] % len(points))
    return list(inside)
