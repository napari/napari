import numpy as np
from copy import copy, deepcopy
from contextlib import contextmanager

from ...util.event import Event
from ...util.misc import ensure_iterable
from ..base import Layer
from vispy.scene.visuals import Mesh, Markers, Compound
from vispy.scene.visuals import Line as VispyLine
from vispy.color import get_color_names
from ._constants import Mode, Box, BACKSPACE, shape_classes, ShapeType
from .shape_list import ShapeList
from .shape_util import create_box, point_to_lines, slice_by_plane
from .shape_models import Rectangle, Ellipse, Line, Path, Polygon


class Shapes(Layer):
    """Shapes layer.

    Parameters
    ----------
    data : list or array
        List of shape data, where each element is an (N, D) array of the
        N vertices of a shape in D dimensions. Can be an 3-dimensional
        array if each shape has the same number of vertices.
    shape_type : string or list
        String of shape shape_type, must be one of "{'line', 'rectangle',
        'ellipse', 'path', 'polygon'}". If a list is supplied it must be
        the same length as the length of `data` and each element will be
        applied to each shape otherwise the same value will be used for all
        shapes.
    edge_width : float or list
        Thickness of lines and edges. If a list is supplied it must be the
        same length as the length of `data` and each element will be
        applied to each shape otherwise the same value will be used for all
        shapes.
    edge_color : str or list
        If string can be any color name recognized by vispy or hex value if
        starting with `#`. If array-like must be 1-dimensional array with 3
        or 4 elements. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape
        otherwise the same value will be used for all shapes.
    face_color : str or list
        If string can be any color name recognized by vispy or hex value if
        starting with `#`. If array-like must be 1-dimensional array with 3
        or 4 elements. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape
        otherwise the same value will be used for all shapes.
    opacity : float or list
        Opacity of the shapes, must be between 0 and 1.
    z_index : int or list
        Specifier of z order priority. Shapes with higher z order are
        displayed ontop of others. If a list is supplied it must be the
        same length as the length of `data` and each element will be
        applied to each shape otherwise the same value will be used for all
        shapes.
    name : str
        Name of the layer.

    Attributes
    ----------
    data : (N, ) list of array
        List of shape data, where each element is an (N, D) array of the
        N vertices of a shape in D dimensions.
    shape_types : (N, ) list of str
        Name of shape type for each shape.
    edge_colors : (N, ) list of str
        Name of edge color for each shape.
    face_colors : (N, ) list of str
        Name of face color for each shape.
    edge_widths : (N, ) list of float
        Edge width for each shape.
    opacities : (N, ) list of float
        Opacity for each shape.
    z_indices : (N, ) list of int
        z-index for each shape.
    edge_width : float
        Thickness of lines and edges of the next shape to be added or the
        currently selected shape.
    edge_color : str
        Color of the edge of the next shape to be added or the currently
        selected shape.
    face_color : str
        Color of the face of the next shape to be added or the currently
        selected shape.
    opacity : float
        Opacity of the next shape to be added or the currently selected shape.
    selected_data : list
        List of currently selected shapes.
    nshapes : int
        Total number of shapes.
    mode : Mode
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        The SELECT mode allows for entire shapes to be selected, moved and
        resized.

        The DIRECT mode allows for shapes to be selected and their individual
        vertices to be moved.

        The VERTEX_INSERT and VERTEX_REMOVE modes allow for individual
        vertices either to be added to or removed from shapes that are already
        selected. Note that shapes cannot be selected in this mode.

        The ADD_RECTANGLE, ADD_ELLIPSE, ADD_LINE, ADD_PATH, and ADD_POLYGON
        modes all allow for their corresponding shape type to be added.

    Extended Summary
    ----------
    _data_dict : Dict of ShapeList
        Dictionary containing all the shape data indexed by slice tuple
    _data_view : ShapeList
        Object containing the currently viewed shape data.
    _nshapes_view : int
        Number of shapes in the current view.
    _mode_history : Mode
        Interactive mode captured on press of <space>.
    _selected_data_history : list
        List of currently selected captured on press of <space>.
    _selected_data_stored : list
        List of selected previously displayed. Used to prevent rerendering the
        same highlighted shapes when no data has changed.
    _selected_box : None | np.ndarray
        `None` if no shapes are selected, otherwise a 10x2 array of vertices of
        the interaction box. The first 8 points are the corners and midpoints
        of the box. The 9th point is the center of the box, and the last point
        is the location of the rotation handle that can be used to rotate the
        box.
    _hover_shape : None | int
        Index of any shape currently hovered over if any. `None` otherwise.
    _hover_shape_stored : None | int
        Index of any shape previously displayed as hovered over if any. `None`
        otherwise. Used to prevent rerendering the same highlighted shapes when
        no data has changed.
    _hover_vertex : None | int
        Index of any vertex currently hovered over if any. `None` otherwise.
    _hover_vertex_stored : None | int
        Index of any vertex previously displayed as hovered over if any. `None`
        otherwise. Used to prevent rerendering the same highlighted shapes when
        no data has changed.
    _moving_shape : None | int
        Index of any shape currently being moved if any. `None` otherwise.
    _moving_vertex : None | int
        Index of any vertex currently being moved if any. `None` otherwise.
    _drag_start : None | np.ndarray
        If a drag has been started and is in progress then a length 2 array of
        the initial coordinates of the drag. `None` otherwise.
    _drag_box : None | np.ndarray
        If a drag box is being created to select shapes then this is a 2x2
        array of the two extreme corners of the drag. `None` otherwise.
    _drag_box_stored : None | np.ndarray
        If a drag box is being created to select shapes then this is a 2x2
        array of the two extreme corners of the drag that have previously been
        rendered. `None` otherwise. Used to prevent rerendering the same
        drag box when no data has changed.
    _is_moving : bool
        Bool indicating if any shapes are currently being moved.
    _is_selecting : bool
        Bool indicating if a drag box is currently being created in order to
        select shapes.
    _is_creating : bool
        Bool indicating if any shapes are currently being created.
    _fixed_aspect : bool
        Bool indicating if aspect ratio of shapes should be preserved on
        resizing.
    _aspect_ratio : float
        Value of aspect ratio to be preserved if `_fixed_aspect` is `True`.
    _fixed_vertex : None | np.ndarray
        If a scaling or rotation is in progress then a length 2 array of the
        coordinates that are remaining fixed during the move. `None` otherwise.
    _fixed_index : int
        If a scaling or rotation is in progress then the index of the vertex of
        the boudning box that is remaining fixed during the move. `None`
        otherwise.
    _update_properties : bool
        Bool indicating if properties are to allowed to update the selected
        shapes when they are changed. Blocking this prevents circular loops
        when shapes are selected and the properties are changed based on that
        selection
    _clipboard : list
        List of shape objects that are to be used during a copy and paste.
    _colors : list
        List of supported vispy color names.
    _vertex_size : float
        Size of the vertices of the shapes and boudning box in Canvas
        coordinates.
    _rotation_handle_length : float
        Length of the rotation handle of the boudning box in Canvas
        coordinates.
    _highlight_color : list
        Length 3 list of color used to highlight shapes and the interaction
        box.
    _highlight_width : float
        Width of the edges used to highlight shapes.
    _input_ndim : int
        Dimensions of shape data.
    """

    _colors = get_color_names()
    _vertex_size = 10
    _rotation_handle_length = 20
    _highlight_color = (0, 0.6, 1)
    _highlight_width = 1.5

    class_keymap = {}

    def __init__(
        self,
        data,
        *,
        shape_type='rectangle',
        edge_width=1,
        edge_color='black',
        face_color='white',
        opacity=0.7,
        z_index=0,
        name=None,
    ):

        # Create a compound visual with the following four subvisuals:
        # Markers: corresponding to the vertices of the interaction box or the
        # shapes that are used for highlights.
        # Lines: The lines of the interaction box used for highlights.
        # Mesh: The mesh of the outlines for each shape used for highlights.
        # Mesh: The actual meshes of the shape faces and edges
        visual = Compound([Markers(), VispyLine(), Mesh(), Mesh()])

        super().__init__(visual, name)

        # Freeze refreshes to prevent drawing before the layer is constructed
        with self.freeze_refresh():

            # The following shape properties are for the new shapes that will
            # be drawn. Each shape has a corresponding property with the
            # value for itself
            if np.isscalar(edge_width):
                self._edge_width = edge_width
            else:
                self._edge_width = 1

            if type(edge_color) is str:
                self._edge_color = edge_color
            else:
                self._edge_color = 'black'

            if type(face_color) is str:
                self._face_color = face_color
            else:
                self._face_color = 'white'

            if np.isscalar(opacity):
                self._opacity = opacity
            else:
                self._opacity = 0.7

            # Add the shape data
            self._input_ndim = None
            self._data_dict = {}
            self._data_view = None

            self.add(
                data,
                shape_type=shape_type,
                edge_width=edge_width,
                edge_color=edge_color,
                face_color=face_color,
                opacity=opacity,
                z_index=z_index,
            )

            # update flags
            self._need_display_update = False
            self._need_visual_update = False

            self._selected_data = []
            self._selected_data_stored = []
            self._selected_data_history = []
            self._selected_box = None

            self._hover_shape = None
            self._hover_shape_stored = None
            self._hover_vertex = None
            self._hover_vertex_stored = None
            self._moving_shape = None
            self._moving_vertex = None

            self._drag_start = None
            self._fixed_vertex = None
            self._fixed_aspect = False
            self._aspect_ratio = 1
            self._is_moving = False
            self._fixed_index = 0
            self._is_selecting = False
            self._drag_box = None
            self._drag_box_stored = None
            self._is_creating = False
            self._clipboard = []

            self._mode = Mode.PAN_ZOOM
            self._mode_history = self._mode
            self._status = self.mode
            self._help = 'enter a selection mode to edit shape properties'

            self.events.add(
                mode=Event,
                edge_width=Event,
                edge_color=Event,
                face_color=Event,
            )

            self.events.deselect.connect(lambda x: self._finish_drawing())
            self.events.face_color.connect(lambda e: self._update_thumbnail())
            self.events.edge_color.connect(lambda e: self._update_thumbnail())

            # Re intitialize indices depending on shape dims
            self._indices = (0,) * (self.ndim - 2) + (
                slice(None, None, None),
                slice(None, None, None),
            )

            # Trigger generation of view slice and thumbnail
            self._set_view_slice()

    @property
    def data(self):
        """list: Each element is an (N, D) array of the vertices of a shape."""
        return self._to_list()

    @data.setter
    def data(self, data):
        self._finish_drawing()
        self._data_dict = {}
        self.add(data, shape_type='rectangle')
        self.events.data()
        self.refresh()

    @property
    def nshapes(self):
        """int: Total number of shapes."""
        nshapes = sum(len(data.shapes) for data in self._data_dict.values())
        return nshapes

    @property
    def _nshapes_view(self):
        """int: Number of shapes in the current view."""
        return len(self._data_view.shapes)

    @property
    def edge_width(self):
        """float: Width of shape edges including lines and paths."""
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width):
        self._edge_width = edge_width
        if self._update_properties:
            index = self.selected_data
            for i in index:
                self._data_view.update_edge_width(i, edge_width)
            self.refresh()
        self.events.edge_width()

    @property
    def edge_color(self):
        """str: color of shape edges including lines and paths."""
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color):
        self._edge_color = edge_color
        if self._update_properties:
            index = self.selected_data
            for i in index:
                self._data_view.update_edge_color(i, edge_color)
            self.refresh()
        self.events.edge_color()

    @property
    def face_color(self):
        """str: color of shape faces."""
        return self._face_color

    @face_color.setter
    def face_color(self, face_color):
        self._face_color = face_color
        if self._update_properties:
            index = self.selected_data
            for i in index:
                self._data_view.update_face_color(i, face_color)
            self.refresh()
        self.events.face_color()

    @property
    def opacity(self):
        """float: Opacity value between 0.0 and 1.0."""
        return self._opacity

    @opacity.setter
    def opacity(self, opacity):
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(
                'opacity must be between 0.0 and 1.0; ' f'got {opacity}'
            )

        self._opacity = opacity
        if self._update_properties:
            index = self.selected_data
            for i in index:
                self._data_view.update_opacity(i, opacity)
            self.refresh()
        self.events.opacity()

    @property
    def shape_types(self):
        """list of str: name of shape type for each shape."""
        shape_types = []
        for d in self._data_dict.values():
            shape_types += d.shape_types
        return shape_types

    @property
    def edge_colors(self):
        """list of str: name of edge color for each shape."""
        edge_colors = []
        for d in self._data_dict.values():
            edge_colors += d.edge_colors
        return edge_colors

    @property
    def face_colors(self):
        """list of str: name of face color for each shape."""
        face_colors = []
        for d in self._data_dict.values():
            face_colors += d.face_colors
        return face_colors

    @property
    def edge_widths(self):
        """list of float: edge width for each shape."""
        edge_widths = []
        for d in self._data_dict.values():
            edge_widths += d.edge_widths
        return edge_widths

    @property
    def opacities(self):
        """list of float: opacity for each shape."""
        opacities = []
        for d in self._data_dict.values():
            opacities += d.opacities
        return opacities

    @property
    def z_indices(self):
        """list of int: z_index for each shape."""
        z_indices = []
        for d in self._data_dict.values():
            z_indices += d.z_indices
        return z_indices

    @property
    def selected_data(self):
        """list: list of currently selected shapes."""
        return self._selected_data

    @selected_data.setter
    def selected_data(self, selected_data):
        self._selected_data = selected_data
        self._selected_box = self.interaction_box(selected_data)

        # Update properties based on selected shapes
        face_colors = list(
            set(
                [
                    self._data_view.shapes[i]._face_color_name
                    for i in selected_data
                ]
            )
        )
        if len(face_colors) == 1:
            face_color = face_colors[0]
            with self.block_update_properties():
                self.face_color = face_color

        edge_colors = list(
            set(
                [
                    self._data_view.shapes[i]._edge_color_name
                    for i in selected_data
                ]
            )
        )
        if len(edge_colors) == 1:
            edge_color = edge_colors[0]
            with self.block_update_properties():
                self.edge_color = edge_color

        edge_width = list(
            set([self._data_view.shapes[i].edge_width for i in selected_data])
        )
        if len(edge_width) == 1:
            edge_width = edge_width[0]
            with self.block_update_properties():
                self.edge_width = edge_width

        opacities = list(
            set([self._data_view.shapes[i].opacity for i in selected_data])
        )
        if len(opacities) == 1:
            opacity = opacities[0]
            with self.block_update_properties():
                self.opacity = opacity

    @property
    def mode(self):
        """MODE: Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        The SELECT mode allows for entire shapes to be selected, moved and
        resized.

        The DIRECT mode allows for shapes to be selected and their individual
        vertices to be moved.

        The VERTEX_INSERT and VERTEX_REMOVE modes allow for individual
        vertices either to be added to or removed from shapes that are already
        selected. Note that shapes cannot be selected in this mode.

        The ADD_RECTANGLE, ADD_ELLIPSE, ADD_LINE, ADD_PATH, and ADD_POLYGON
        modes all allow for their corresponding shape type to be added.
        """
        return str(self._mode)

    @mode.setter
    def mode(self, mode):
        if isinstance(mode, str):
            mode = Mode(mode)

        if mode == self._mode:
            return
        old_mode = self._mode
        if mode == Mode.PAN_ZOOM:
            self.cursor = 'standard'
            self.interactive = True
            self.help = 'enter a selection mode to edit shape properties'
        elif mode in [Mode.SELECT, Mode.DIRECT]:
            self.cursor = 'pointing'
            self.interactive = False
            self.help = (
                'hold <space> to pan/zoom, '
                f'press <{BACKSPACE}> to remove selected'
            )
        elif mode in [Mode.VERTEX_INSERT, Mode.VERTEX_REMOVE]:
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom'
        elif mode in [Mode.ADD_RECTANGLE, Mode.ADD_ELLIPSE, Mode.ADD_LINE]:
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom'
        elif mode in [Mode.ADD_PATH, Mode.ADD_POLYGON]:
            self.cursor = 'cross'
            self.interactive = False
            self.help = (
                'hold <space> to pan/zoom, ' 'press <esc> to finish drawing'
            )
        else:
            raise ValueError("Mode not recongnized")

        self.status = str(mode)
        self._mode = mode

        draw_modes = [
            Mode.SELECT,
            Mode.DIRECT,
            Mode.VERTEX_INSERT,
            Mode.VERTEX_REMOVE,
        ]

        self.events.mode(mode=mode)
        if not (mode in draw_modes and old_mode in draw_modes):
            self._finish_drawing()
        self.refresh()

    def _get_shape(self):
        """Determines the shape of the vertices of the shape data."""
        if len(self._data_view._vertices) == 0:
            slice_shape = tuple([1, 1])
        else:
            slice_shape = tuple(np.max(self._data_view._vertices, axis=0) + 1)

        slice_keys = list(self._data_dict.keys())
        max_val = np.array(slice_keys).max(axis=0)
        return tuple(max_val) + slice_shape

    @property
    def range(self):
        """list of 3-tuple: ranges for slicing given by (min, max, step)."""
        if len(self._data_view._vertices) == 0:
            maxs = [1, 1]
            mins = [0, 0]
        else:
            maxs = np.max(self._data_view._vertices, axis=0) + 1
            mins = np.min(self._data_view._vertices, axis=0)

        slice_keys = list(self._data_dict.keys())
        min_val = np.array(slice_keys).min(axis=0)
        max_val = np.array(slice_keys).max(axis=0)

        mins = tuple(min_val) + tuple(mins)
        maxs = tuple(max_val) + tuple(maxs)

        return tuple((min, max, 1) for min, max in zip(mins, maxs))

    def add(
        self,
        data,
        *,
        shape_type='rectangle',
        edge_width=None,
        edge_color=None,
        face_color=None,
        opacity=None,
        z_index=None,
    ):
        """Add shapes to the current layer.

        Parameters
        ----------
        data : list or array
            List of shape data, where each element is an (N, D) array of the
            N vertices of a shape in D dimensions. Can be an 3-dimensional
            array if each shape has the same number of vertices.
        shape_type : string | list
            String of shape shape_type, must be one of "{'line', 'rectangle',
            'ellipse', 'path', 'polygon'}". If a list is supplied it must be
            the same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        edge_width : float | list
            thickness of lines and edges. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        edge_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        face_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each shape
            otherwise the same value will be used for all shapes.
        opacity : float | list
            Opacity of the shapes, must be between 0 and 1.
        z_index : int | list
            Specifier of z order priority. Shapes with higher z order are
            displayed ontop of others. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each shape otherwise the same value will be used for all
            shapes.
        """
        if edge_width is None:
            edge_width = self.edge_width
        if edge_color is None:
            edge_color = self.edge_color
        if face_color is None:
            face_color = self.face_color
        if opacity is None:
            opacity = self.opacity
        if self._data_view is not None:
            z_index = z_index or max(self._data_view._z_index, default=-1) + 1
        else:
            z_index = z_index or 0

        if len(data) > 0:
            if np.array(data[0]).ndim == 1:
                # If a single array for a shape has been passed turn into list
                data = [data]

            # Turn input arguments into iterables
            shape_inputs = zip(
                data,
                ensure_iterable(shape_type),
                ensure_iterable(edge_width),
                ensure_iterable(edge_color, color=True),
                ensure_iterable(face_color, color=True),
                ensure_iterable(opacity),
                ensure_iterable(z_index),
            )

            for d, st, ew, ec, fc, o, z in shape_inputs:
                shape_cls = shape_classes[ShapeType(st)]

                # Slice data by 2D plane.
                slice_key, data_2D = slice_by_plane(d)
                # A False slice_key means the shape is invalid as it is not
                # confined to a single plane
                if slice_key is not False:
                    shape = shape_cls(
                        data_2D,
                        edge_width=ew,
                        edge_color=ec,
                        face_color=fc,
                        opacity=o,
                        z_index=z,
                    )
                    # If data is being drawn in gui it will already be 2D and
                    # so should just be added to the current ShapeList
                    if slice_key == ():
                        if len(self._data_dict) == 0:
                            # If input dim not initialized, set value
                            if self._input_ndim is None:
                                self._input_ndim = 2
                            self._data_dict[slice_key] = ShapeList()
                            self._data_view = self._data_dict[slice_key]
                        self._data_view.add(shape)
                    elif slice_key in self._data_dict:
                        self._data_dict[slice_key].add(shape)
                    else:
                        # If input dim not initialized, set value
                        if self._input_ndim is None:
                            self._input_ndim = 2 + len(slice_key)
                        # Check shape has correct dimensions
                        if self._input_ndim == 2 + len(slice_key):
                            self._data_dict[slice_key] = ShapeList()
                            self._data_dict[slice_key].add(shape)
                        else:
                            raise ValueError(
                                'all shapes must have the same dimension'
                            )

        # If input_ndim has not been set, default to 2
        if self._input_ndim is None:
            # If data was empty 3D array
            if len(data) == 0 and type(data) == np.ndarray:
                self._input_ndim = data.shape[-1]
            else:
                self._input_ndim = 2

        # If _data_view has not yet been definied,
        # set the currently viewed slice to top slice
        if self._data_view is None:
            init_index = (0,) * (self._input_ndim - 2)
            if init_index not in self._data_dict:
                self._data_dict[init_index] = ShapeList()
            self._data_view = self._data_dict[init_index]

        self._update_thumbnail()

    def _set_view_slice(self):
        """Set the view given the slicing indices."""
        with self.freeze_refresh():
            slice_key = self.indices[:-2]
            if slice_key not in self._data_dict:
                self._data_dict[slice_key] = ShapeList()
            if not self._data_view == self._data_dict[slice_key]:
                self._data_view = self._data_dict[slice_key]
                # If data is changed unselect all shapes
                self._finish_drawing()

        z_order = self._data_view._mesh.triangles_z_order
        faces = self._data_view._mesh.triangles[z_order]
        colors = self._data_view._mesh.triangles_colors[z_order]
        vertices = self._data_view._mesh.vertices[:, ::-1]
        if len(faces) == 0:
            self._node._subvisuals[3].set_data(vertices=None, faces=None)
        else:
            self._node._subvisuals[3].set_data(
                vertices=vertices, faces=faces, face_colors=colors
            )
        self._need_visual_update = True
        self._set_highlight(force=True)
        self._update()

    def interaction_box(self, index):
        """Create the interaction box around a shape or list of shapes.
        If a single index is passed then the boudning box will be inherited
        from that shapes interaction box. If list of indices is passed it will
        be computed directly.

        Parameters
        ----------
        index : int | list
            Index of a single shape, or a list of shapes around which to
            construct the interaction box

        Returns
        ----------
        box : np.ndarray
            10x2 array of vertices of the interaction box. The first 8 points
            are the corners and midpoints of the box in clockwise order
            starting in the upper-left corner. The 9th point is the center of
            the box, and the last point is the location of the rotation handle
            that can be used to rotate the box
        """
        if isinstance(index, (list, np.ndarray)):
            if len(index) == 0:
                box = None
            elif len(index) == 1:
                box = copy(self._data_view.shapes[index[0]]._box)
            else:
                indices = np.isin(self._data_view._index, index)
                box = create_box(self._data_view._vertices[indices])
        else:
            box = copy(self._data_view.shapes[index]._box)

        if box is not None:
            rot = box[Box.TOP_CENTER]
            length_box = np.linalg.norm(
                box[Box.BOTTOM_LEFT] - box[Box.TOP_LEFT]
            )
            if length_box > 0:
                r = self._rotation_handle_length * self.scale_factor
                rot = (
                    rot
                    - r
                    * (box[Box.BOTTOM_LEFT] - box[Box.TOP_LEFT])
                    / length_box
                )
            box = np.append(box, [rot], axis=0)

        return box

    def _outline_shapes(self):
        """Find outlines of any selected or hovered shapes.

        Returns
        ----------
        vertices : None | np.ndarray
            Nx2 array of any vertices of outline or None
        triangles : None | np.ndarray
            Mx3 array of any indices of vertices for triangles of outline or
            None
        """
        if self._hover_shape is not None or len(self.selected_data) > 0:
            if len(self.selected_data) > 0:
                index = copy(self.selected_data)
                if self._hover_shape is not None:
                    if self._hover_shape in index:
                        pass
                    else:
                        index.append(self._hover_shape)
                index.sort()
            else:
                index = self._hover_shape

            centers, offsets, triangles = self._data_view.outline(index)
            vertices = centers + (
                self.scale_factor * self._highlight_width * offsets
            )
            vertices = vertices[:, ::-1]
        else:
            vertices = None
            triangles = None

        return vertices, triangles

    def _compute_vertices_and_box(self):
        """Compute location of highlight vertices and box for rendering.

        Returns
        ----------
        vertices : np.ndarray
            Nx2 array of any vertices to be rendered as Markers
        face_color : str
            String of the face color of the Markers
        edge_color : str
            String of the edge color of the Markers and Line for the box
        pos : np.ndarray
            Nx2 array of vertices of the box that will be rendered using a
            Vispy Line
        width : float
            Width of the box edge
        """
        if len(self.selected_data) > 0:
            if self._mode == Mode.SELECT:
                # If in select mode just show the interaction boudning box
                # including its vertices and the rotation handle
                box = self._selected_box[Box.WITH_HANDLE]
                if self._hover_shape is None:
                    face_color = 'white'
                elif self._hover_vertex is None:
                    face_color = 'white'
                else:
                    face_color = self._highlight_color
                edge_color = self._highlight_color
                vertices = box[:, ::-1]
                # Use a subset of the vertices of the interaction_box to plot
                # the line around the edge
                pos = box[Box.LINE_HANDLE][:, ::-1]
                width = 1.5
            elif self._mode in (
                [
                    Mode.DIRECT,
                    Mode.ADD_PATH,
                    Mode.ADD_POLYGON,
                    Mode.ADD_RECTANGLE,
                    Mode.ADD_ELLIPSE,
                    Mode.ADD_LINE,
                    Mode.VERTEX_INSERT,
                    Mode.VERTEX_REMOVE,
                ]
            ):
                # If in one of these mode show the vertices of the shape itself
                inds = np.isin(self._data_view._index, self.selected_data)
                vertices = self._data_view._vertices[inds][:, ::-1]
                # If currently adding path don't show box over last vertex
                if self._mode == Mode.ADD_PATH:
                    vertices = vertices[:-1]

                if self._hover_shape is None:
                    face_color = 'white'
                elif self._hover_vertex is None:
                    face_color = 'white'
                else:
                    face_color = self._highlight_color
                edge_color = self._highlight_color
                pos = None
                width = 0
            else:
                # Otherwise show nothing
                vertices = np.empty((0, 2))
                face_color = 'white'
                edge_color = 'white'
                pos = None
                width = 0
        elif self._is_selecting:
            # If currently dragging a selection box just show an outline of
            # that box
            vertices = np.empty((0, 2))
            edge_color = self._highlight_color
            face_color = 'white'
            box = create_box(self._drag_box)
            width = 1.5
            # Use a subset of the vertices of the interaction_box to plot
            # the line around the edge
            pos = box[Box.LINE][:, ::-1]
        else:
            # Otherwise show nothing
            vertices = np.empty((0, 2))
            face_color = 'white'
            edge_color = 'white'
            pos = None
            width = 0

        return vertices, face_color, edge_color, pos, width

    def _set_highlight(self, force=False):
        """Render highlights of shapes.

        Includes boundaries, vertices, interaction boxes, and the drag
        selection box when appropriate.

        Parameters
        ----------
        force : bool
            Bool that forces a redraw to occur when `True`
        """
        # Check if any shape or vertex ids have changed since last call
        if (
            self.selected_data == self._selected_data_stored
            and self._hover_shape == self._hover_shape_stored
            and self._hover_vertex == self._hover_vertex_stored
            and np.all(self._drag_box == self._drag_box_stored)
        ) and not force:
            return
        self._selected_data_stored = copy(self.selected_data)
        self._hover_shape_stored = copy(self._hover_shape)
        self._hover_vertex_stored = copy(self._hover_vertex)
        self._drag_box_stored = copy(self._drag_box)

        # Compute the vertices and faces of any shape outlines
        vertices, faces = self._outline_shapes()
        self._node._subvisuals[2].set_data(
            vertices=vertices, faces=faces, color=self._highlight_color
        )

        # Compute the location and properties of the vertices and box that
        # need to get rendered
        (
            vertices,
            face_color,
            edge_color,
            pos,
            width,
        ) = self._compute_vertices_and_box()
        self._node._subvisuals[0].set_data(
            vertices,
            size=self._vertex_size,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=1.5,
            symbol='square',
            scaling=False,
        )
        self._node._subvisuals[1].set_data(
            pos=pos, color=edge_color, width=width
        )

    def _finish_drawing(self):
        """Reset properties used in shape drawing."""
        index = copy(self._moving_shape)
        self._is_moving = False
        self.selected_data = []
        self._drag_start = None
        self._drag_box = None
        self._is_selecting = False
        self._fixed_vertex = None
        self._moving_shape = None
        self._moving_vertex = None
        self._hover_shape = None
        self._hover_vertex = None
        if self._is_creating is True and self._mode == Mode.ADD_PATH:
            vertices = self._data_view._vertices[
                self._data_view._index == index
            ]
            if len(vertices) <= 2:
                self._data_view.remove(index)
            else:
                self._data_view.edit(index, vertices[:-1])
        if self._is_creating is True and self._mode == Mode.ADD_POLYGON:
            vertices = self._data_view._vertices[
                self._data_view._index == index
            ]
            if len(vertices) <= 2:
                self._data_view.remove(index)
        self._is_creating = False
        self.refresh()
        self._update_thumbnail()

    def _update_thumbnail(self):
        """Update thumbnail with current points and colors."""
        # calculate min vals for the vertices and pad with 0.5
        # the offset is needed to ensure that the top left corner of the shapes
        # corresponds to the top left corner of the thumbnail
        offset = np.array([self.range[-2][0], self.range[-1][0]]) - 0.5
        # calculate range of values for the vertices and pad with 1
        # padding ensures the entire shape can be represented in the thumbnail
        # without getting clipped
        shape = np.ceil(
            [
                self.range[-2][1] - self.range[-2][0] + 1,
                self.range[-1][1] - self.range[-1][0] + 1,
            ]
        ).astype(int)
        zoom_factor = np.divide(self._thumbnail_shape[:2], shape).min()

        colormapped = self._data_view.to_colors(
            colors_shape=self._thumbnail_shape[:2],
            zoom_factor=zoom_factor,
            offset=offset,
        )

        self.thumbnail = colormapped

    def remove_selected(self):
        """Remove any selected shapes."""
        to_remove = sorted(self.selected_data, reverse=True)
        for index in to_remove:
            self._data_view.remove(index)
        self.selected_data = []
        shape, vertex = self.get_value(self.coordinates[-2:])
        self._hover_shape = shape
        self._hover_vertex = vertex
        self.status = self.get_message(self.coordinates[-2:], shape, vertex)
        self._finish_drawing()

    def _rotate_box(self, angle, center=[0, 0]):
        """Perfrom a rotation on the selected box.

        Parameters
        ----------
        angle : float
            angle specifying rotation of shapes in degrees.
        center : list
            coordinates of center of rotation.
        """
        theta = np.radians(angle)
        transform = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )
        box = self._selected_box - center
        self._selected_box = box @ transform.T + center

    def _scale_box(self, scale, center=[0, 0]):
        """Perfrom a scaling on the selected box.

        Parameters
        ----------
        scale : float, list
            scalar or list specifying rescaling of shape.
        center : list
            coordinates of center of rotation.
        """
        if not isinstance(scale, (list, np.ndarray)):
            scale = [scale, scale]
        box = self._selected_box - center
        box = np.array(box * scale)
        if not np.all(box[Box.TOP_CENTER] == box[Box.HANDLE]):
            r = self._rotation_handle_length * self.scale_factor
            handle_vec = box[Box.HANDLE] - box[Box.TOP_CENTER]
            cur_len = np.linalg.norm(handle_vec)
            box[Box.HANDLE] = box[Box.TOP_CENTER] + r * handle_vec / cur_len
        self._selected_box = box + center

    def _transform_box(self, transform, center=[0, 0]):
        """Perfrom a linear transformation on the selected box.

        Parameters
        ----------
        transform : np.ndarray
            2x2 array specifying linear transform.
        center : list
            coordinates of center of rotation.
        """
        box = self._selected_box - center
        box = box @ transform.T
        if not np.all(box[Box.TOP_CENTER] == box[Box.HANDLE]):
            r = self._rotation_handle_length * self.scale_factor
            handle_vec = box[Box.HANDLE] - box[Box.TOP_CENTER]
            cur_len = np.linalg.norm(handle_vec)
            box[Box.HANDLE] = box[Box.TOP_CENTER] + r * handle_vec / cur_len
        self._selected_box = box + center

    def get_value(self, coord):
        """Determine if any shape at given coord using triangle meshes.

        Parameters
        ----------
        coord : sequence of float
            Image coordinates to check if any shapes are at.

        Returns
        ----------
        shape : int | None
            Index of shape if any that is at the coordinates. Returns `None`
            if no shape is found.
        vertex : int | None
            Index of vertex if any that is at the coordinates. Returns `None`
            if no vertex is found.
        """
        # Check selected shapes
        if len(self.selected_data) > 0:
            if self._mode == Mode.SELECT:
                # Check if inside vertex of interaction box or rotation handle
                box = self._selected_box[Box.WITH_HANDLE]
                distances = abs(box - coord[:2])

                # Get the vertex sizes
                sizes = self._vertex_size * self.scale_factor / 2

                # Check if any matching vertices
                matches = np.all(distances <= sizes, axis=1).nonzero()
                if len(matches[0]) > 0:
                    return self.selected_data[0], matches[0][-1]
            elif self._mode in (
                [Mode.DIRECT, Mode.VERTEX_INSERT, Mode.VERTEX_REMOVE]
            ):
                # Check if inside vertex of shape
                inds = np.isin(self._data_view._index, self.selected_data)
                vertices = self._data_view._vertices[inds]
                distances = abs(vertices - coord[:2])

                # Get the vertex sizes
                sizes = self._vertex_size * self.scale_factor / 2

                # Check if any matching vertices
                matches = np.all(distances <= sizes, axis=1).nonzero()[0]
                if len(matches) > 0:
                    index = inds.nonzero()[0][matches[-1]]
                    shape = self._data_view._index[index]
                    _, idx = np.unique(
                        self._data_view._index, return_index=True
                    )
                    return shape, index - idx[shape]

        # Check if mouse inside shape
        shape = self._data_view.inside(coord)
        return shape, None

    def get_message(self, coord, shape, vertex):
        """Generate a string based on the coordinates hover values

        Parameters
        ----------
        coord : sequence of int
            Position of mouse cursor in image coordinates.
        shape : int | None
            Index of shape if any to be highlighted.
        vertex : int | None
            Index of vertex if any to be highlighted.
        Returns
        ----------
        msg : string
            String containing a message that can be used as a status update.
        """
        int_coord = np.round(coord).astype(int)
        msg = f'{int_coord}, {self.name}'
        if shape is not None:
            msg = msg + ', shape ' + str(shape)
            if vertex is not None:
                msg = msg + ', vertex ' + str(vertex)
        return msg

    def move_to_front(self):
        """Moves selected objects to be displayed in front of all others."""
        if len(self.selected_data) == 0:
            return
        new_z_index = max(self._data_view._z_index) + 1
        for index in self.selected_data:
            self._data_view.update_z_index(index, new_z_index)
        self.refresh()

    def move_to_back(self):
        """Moves selected objects to be displayed behind all others."""
        if len(self.selected_data) == 0:
            return
        new_z_index = min(self._data_view._z_index) - 1
        for index in self.selected_data:
            self._data_view.update_z_index(index, new_z_index)
        self.refresh()

    def _copy_data(self):
        """Copy selected shapes to clipboard."""
        self._clipboard = [
            deepcopy(self._data_view.shapes[i]) for i in self._selected_data
        ]

    def _paste_data(self):
        """Paste any shapes from clipboard and then selects them."""
        cur_shapes = self._nshapes_view
        for s in self._clipboard:
            self._data_view.add(deepcopy(s))
        self.selected_data = list(
            range(cur_shapes, cur_shapes + len(self._clipboard))
        )
        self.move_to_front()

    def _move(self, coord):
        """Moves object at given mouse position and set of indices.

        Parameters
        ----------
        coord : sequence of two int
            Position of mouse cursor in image coordinates.
        """
        vertex = self._moving_vertex
        if self._mode in (
            [Mode.SELECT, Mode.ADD_RECTANGLE, Mode.ADD_ELLIPSE, Mode.ADD_LINE]
        ):
            if len(self.selected_data) > 0:
                self._is_moving = True
                if vertex is None:
                    # Check where dragging box from to move whole object
                    if self._drag_start is None:
                        center = self._selected_box[Box.CENTER]
                        self._drag_start = coord - center
                    center = self._selected_box[Box.CENTER]
                    shift = coord - center - self._drag_start
                    for index in self.selected_data:
                        self._data_view.shift(index, shift)
                    self._selected_box = self._selected_box + shift
                    self.refresh()
                elif vertex < Box.LEN:
                    # Corner / edge vertex is being dragged so resize object
                    box = self._selected_box
                    if self._fixed_vertex is None:
                        self._fixed_index = (vertex + 4) % Box.LEN
                        self._fixed_vertex = box[self._fixed_index]

                    size = (
                        box[(self._fixed_index + 4) % Box.LEN]
                        - box[self._fixed_index]
                    )
                    offset = box[Box.HANDLE] - box[Box.CENTER]
                    offset = offset / np.linalg.norm(offset)
                    offset_perp = np.array([offset[1], -offset[0]])

                    fixed = self._fixed_vertex
                    new = copy(coord)

                    if self._fixed_aspect and self._fixed_index % 2 == 0:
                        if (new - fixed)[0] == 0:
                            ratio = 1
                        else:
                            ratio = abs((new - fixed)[1] / (new - fixed)[0])
                        if ratio > self._aspect_ratio:
                            r = self._aspect_ratio / ratio
                            new[1] = fixed[1] + (new[1] - fixed[1]) * r
                        else:
                            r = ratio / self._aspect_ratio
                            new[0] = fixed[0] + (new[0] - fixed[0]) * r

                    if size @ offset == 0:
                        dist = 1
                    else:
                        dist = ((new - fixed) @ offset) / (size @ offset)

                    if size @ offset_perp == 0:
                        dist_perp = 1
                    else:
                        dist_perp = ((new - fixed) @ offset_perp) / (
                            size @ offset_perp
                        )

                    if self._fixed_index % 2 == 0:
                        # corner selected
                        scale = np.array([dist_perp, dist])
                    elif self._fixed_index % 4 == 3:
                        # top selected
                        scale = np.array([1, dist])
                    else:
                        # side selected
                        scale = np.array([dist_perp, 1])

                    # prevent box from shrinking below a threshold size
                    threshold = self._vertex_size * self.scale_factor / 8
                    scale[abs(scale * size[[1, 0]]) < threshold] = 1

                    # check orientation of box
                    angle = -np.arctan2(offset[0], -offset[1])
                    c, s = np.cos(angle), np.sin(angle)
                    if angle == 0:
                        for index in self.selected_data:
                            self._data_view.scale(
                                index, scale, center=self._fixed_vertex
                            )
                        self._scale_box(scale, center=self._fixed_vertex)
                    else:
                        rotation = np.array([[c, s], [-s, c]])
                        scale_mat = np.array([[scale[0], 0], [0, scale[1]]])
                        inv_rot = np.array([[c, -s], [s, c]])
                        transform = rotation @ scale_mat @ inv_rot
                        for index in self.selected_data:
                            self._data_view.shift(index, -self._fixed_vertex)
                            self._data_view.transform(index, transform)
                            self._data_view.shift(index, self._fixed_vertex)
                        self._transform_box(
                            transform, center=self._fixed_vertex
                        )
                    self.refresh()
                elif vertex == 8:
                    # Rotation handle is being dragged so rotate object
                    handle = self._selected_box[Box.HANDLE]
                    if self._drag_start is None:
                        self._fixed_vertex = self._selected_box[Box.CENTER]
                        offset = handle - self._fixed_vertex
                        self._drag_start = -np.degrees(
                            np.arctan2(offset[0], -offset[1])
                        )

                    new_offset = coord - self._fixed_vertex
                    new_angle = -np.degrees(
                        np.arctan2(new_offset[0], -new_offset[1])
                    )
                    fixed_offset = handle - self._fixed_vertex
                    fixed_angle = -np.degrees(
                        np.arctan2(fixed_offset[0], -fixed_offset[1])
                    )

                    if np.linalg.norm(new_offset) < 1:
                        angle = 0
                    elif self._fixed_aspect:
                        angle = np.round(new_angle / 45) * 45 - fixed_angle
                    else:
                        angle = new_angle - fixed_angle

                    for index in self.selected_data:
                        self._data_view.rotate(
                            index, angle, center=self._fixed_vertex
                        )
                    self._rotate_box(angle, center=self._fixed_vertex)
                    self.refresh()
            else:
                self._is_selecting = True
                if self._drag_start is None:
                    self._drag_start = coord
                self._drag_box = np.array([self._drag_start, coord])
                self._set_highlight()
        elif self._mode in [Mode.DIRECT, Mode.ADD_PATH, Mode.ADD_POLYGON]:
            if len(self.selected_data) > 0:
                if vertex is not None:
                    self._is_moving = True
                    index = self._moving_shape
                    shape_type = type(self._data_view.shapes[index])
                    if shape_type == Ellipse:
                        # DIRECT vertex moving of ellipse not implemented
                        pass
                    else:
                        if shape_type == Rectangle:
                            new_type = Polygon
                        else:
                            new_type = None
                        indices = self._data_view._index == index
                        vertices = self._data_view._vertices[indices]
                        vertices[vertex] = coord
                        self._data_view.edit(
                            index, vertices, new_type=new_type
                        )
                        shapes = self.selected_data
                        self._selected_box = self.interaction_box(shapes)
                        self.refresh()
            else:
                self._is_selecting = True
                if self._drag_start is None:
                    self._drag_start = coord
                self._drag_box = np.array([self._drag_start, coord])
                self._set_highlight()
        elif self._mode in [Mode.VERTEX_INSERT, Mode.VERTEX_REMOVE]:
            if len(self.selected_data) > 0:
                pass
            else:
                self._is_selecting = True
                if self._drag_start is None:
                    self._drag_start = coord
                self._drag_box = np.array([self._drag_start, coord])
                self._set_highlight()

    def to_xml_list(self, shape_type=None):
        """Convert the shapes to a list of svg xml elements.

        Z ordering of the shapes will be taken into account.

        Parameters
        ----------
        shape_type : {'line', 'rectangle', 'ellipse', 'path', 'polygon'},
            optional
            String of which shape types should to be included in the xml.

        Returns
        ----------
        xml : list
            List of xml elements defining each shape according to the
            svg specification
        """
        return self._data_view.to_xml_list(shape_type=shape_type)

    def to_masks(self, mask_shape=None, shape_type=None):
        """Return an array of binary masks, one for each shape.

        Parameters
        ----------
        mask_shape : np.ndarray | tuple | None
            tuple defining shape of mask to be generated. If non specified,
            takes the max of all the vertiecs
        shape_type : {'line', 'rectangle', 'ellipse', 'path', 'polygon'} |
                     None, optional
            String of shape type to be included.

        Returns
        ----------
        masks : np.ndarray
            Array where there is one binary mask for each shape
        """
        if mask_shape is None:
            mask_shape = self.shape

        mask_shape = np.ceil(mask_shape).astype('int')

        if self.ndim == 2:
            # For 2D shapes just convert current view to masks and
            # broadcast across sliced dimensions
            slices = self._data_view.to_masks(
                mask_shape=mask_shape[-2:], shape_type=shape_type
            )
            masks = [np.broadcast_to(m, mask_shape) for m in slices]
        else:
            # For nD insert each keyed slice into correct place in volume
            masks = []
            for slice_key, data in self._data_dict.items():
                if len(slice_key) > 0:
                    slices = data.to_masks(
                        mask_shape=mask_shape[-2:], shape_type=shape_type
                    )
                    for m in slices:
                        vol = np.zeros(mask_shape)
                        vol[slice_key] = m
                        masks.append(vol)
        if len(masks) == 0:
            masks = np.array(masks)
        else:
            masks = np.stack(masks, axis=0)
        return masks

    def to_labels(self, labels_shape=None, shape_type=None):
        """Return an integer labels image.

        Parameters
        ----------
        labels_shape : np.ndarray | tuple | None
            Tuple defining shape of labels image to be generated. If non
            specified, takes the max of all the vertiecs
        shape_type : {'line', 'rectangle', 'ellipse', 'path', 'polygon'} |
                     None, optional
            String of shape type to be included.

        Returns
        ----------
        labels : np.ndarray
            Integer array where each value is either 0 for background or an
            integer up to N for points inside the shape at the index value - 1.
            For overlapping shapes z-ordering will be respected.
        """
        if labels_shape is None:
            labels_shape = self.shape

        labels_shape = np.ceil(labels_shape).astype('int')

        if self.ndim == 2:
            # For 2D shapes convert current view to labels
            # and broadcast across sliced dimensions
            labels = self._data_view.to_labels(
                labels_shape=labels_shape[-2:], shape_type=shape_type
            )
            labels = np.broadcast_to(labels, labels_shape)
        else:
            # For nD insert each keyed slice into correct place in volume
            # and increment integer label of shape
            labels = np.zeros(labels_shape)
            nshapes = 0
            for slice_key, data in self._data_dict.items():
                slices = data.to_labels(
                    labels_shape=labels_shape[-2:], shape_type=shape_type
                )
                slices[slices > 0] += nshapes
                labels[slice_key] = slices
                nshapes += len(data.shapes)
        return labels

    def _to_list(self, shape_type=None):
        """Return the vertex data assoicated with the shapes as a list.

        Parameters
        ----------
        shape_type : {'line', 'rectangle', 'ellipse', 'path', 'polygon'} |
                     None, optional
            String of shape type to be included.

        Returns
        ----------
        data : list
            List of shape data where each element of the list is an
            `np.ndarray` corresponding to one shape
        """

        if self.ndim == 2:
            # For 2D shapes convert current view to a list
            data = self._data_view.to_list(shape_type=shape_type)
        else:
            # For nD insert each slice_key into shape indices in list
            data = []
            for slice_key, d in self._data_dict.items():
                shapes = d.to_list(shape_type=shape_type)
                for s in shapes:
                    slice_keys = np.tile(slice_key, (len(s), 1))
                    full_shape = np.concatenate((slice_keys, s), axis=1)
                    data.append(full_shape)
        return data

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if event.pos is None:
            return
        self.position = tuple(event.pos)
        coord = self.coordinates[-2:]
        shift = 'Shift' in event.modifiers

        if self._mode == Mode.PAN_ZOOM:
            # If in pan/zoom mode do nothing
            pass
        elif self._mode in [Mode.SELECT, Mode.DIRECT]:
            if not self._is_moving and not self._is_selecting:
                shape, vertex = self.get_value(coord)
                self._moving_shape = shape
                self._moving_vertex = vertex
                if vertex is None:
                    if shift and shape is not None:
                        if shape in self.selected_data:
                            self.selected_data.remove(shape)
                            shapes = self.selected_data
                            self._selected_box = self.interaction_box(shapes)
                        else:
                            self.selected_data.append(shape)
                            shapes = self.selected_data
                            self._selected_box = self.interaction_box(shapes)
                    elif shape is not None:
                        if shape not in self.selected_data:
                            self.selected_data = [shape]
                    else:
                        self.selected_data = []
                self._set_highlight()
                self.status = self.get_message(coord, shape, vertex)
        elif self._mode in (
            [Mode.ADD_RECTANGLE, Mode.ADD_ELLIPSE, Mode.ADD_LINE]
        ):
            # Start drawing a rectangle / ellipse / line
            size = self._vertex_size * self.scale_factor / 4
            if self._mode == Mode.ADD_RECTANGLE:
                data = np.array([coord, coord + size])
                shape_type = 'rectangle'
            elif self._mode == Mode.ADD_ELLIPSE:
                data = np.array([coord + size / 2, [size, size]])
                shape_type = 'ellipse'
            elif self._mode == Mode.ADD_LINE:
                data = np.array([coord, coord + size])
                shape_type = 'line'
            self.add(data, shape_type=shape_type)
            self.selected_data = [self._nshapes_view - 1]
            ind = 4
            self._moving_shape = self.selected_data[0]
            self._moving_vertex = ind
            self._hover_shape = self.selected_data[0]
            self._hover_vertex = ind
            self._is_creating = True
            self._set_highlight()
            self.refresh()
        elif self._mode in [Mode.ADD_PATH, Mode.ADD_POLYGON]:
            if self._is_creating is False:
                # Start drawing a path
                data = np.array([coord, coord])
                self.add(data, shape_type='path')
                self.selected_data = [self._nshapes_view - 1]
                ind = 1
                self._moving_shape = self.selected_data[0]
                self._moving_vertex = ind
                self._hover_shape = self.selected_data[0]
                self._hover_vertex = ind
                self._is_creating = True
                self._set_highlight()
            else:
                # Add to an existing path or polygon
                index = self._moving_shape
                if self._mode == Mode.ADD_POLYGON:
                    new_type = Polygon
                else:
                    new_type = None
                vertices = self._data_view._vertices[
                    self._data_view._index == index
                ]
                vertices = np.concatenate((vertices, [coord]), axis=0)
                # Change the selected vertex
                self._moving_vertex = self._moving_vertex + 1
                self._hover_vertex = self._hover_vertex + 1
                self._data_view.edit(index, vertices, new_type=new_type)
                self._selected_box = self.interaction_box(self.selected_data)
            self.status = self.get_message(
                coord, self._hover_shape, self._hover_vertex
            )
        elif self._mode == Mode.VERTEX_INSERT:
            if len(self.selected_data) == 0:
                # If none selected return immediately
                return

            all_lines = np.empty((0, 2, 2))
            all_lines_shape = np.empty((0, 2), dtype=int)
            for index in self.selected_data:
                shape_type = type(self._data_view.shapes[index])
                if shape_type == Ellipse:
                    # Adding vertex to ellipse not implemented
                    pass
                else:
                    vertices = self._data_view._vertices[
                        self._data_view._index == index
                    ]
                    # Find which edge new vertex should inserted along
                    closed = shape_type != Path
                    n = len(vertices)
                    if closed:
                        lines = np.array(
                            [
                                [vertices[i], vertices[(i + 1) % n]]
                                for i in range(n)
                            ]
                        )
                    else:
                        lines = np.array(
                            [
                                [vertices[i], vertices[i + 1]]
                                for i in range(n - 1)
                            ]
                        )
                    all_lines = np.append(all_lines, lines, axis=0)
                    indices = np.array(
                        [np.repeat(index, len(lines)), list(range(len(lines)))]
                    ).T
                    all_lines_shape = np.append(
                        all_lines_shape, indices, axis=0
                    )
            if len(all_lines) == 0:
                # No appropriate shapes found
                return
            ind, loc = point_to_lines(coord, all_lines)
            index = all_lines_shape[ind][0]
            ind = all_lines_shape[ind][1] + 1
            shape_type = type(self._data_view.shapes[index])
            if shape_type == Line:
                # Adding vertex to line turns it into a path
                new_type = Path
            elif shape_type == Rectangle:
                # Adding vertex to rectangle turns it into a polygon
                new_type = Polygon
            else:
                new_type = None
            closed = shape_type != Path
            vertices = self._data_view._vertices[
                self._data_view._index == index
            ]
            if closed is not True:
                if int(ind) == 1 and loc < 0:
                    ind = 0
                elif int(ind) == len(vertices) - 1 and loc > 1:
                    ind = ind + 1

            vertices = np.insert(vertices, ind, [coord], axis=0)
            with self.freeze_refresh():
                self._data_view.edit(index, vertices, new_type=new_type)
                self._selected_box = self.interaction_box(self.selected_data)
            shape, vertex = self.get_value(coord)
            self._hover_shape = shape
            self._hover_vertex = vertex
            self.refresh()
            self.status = self.get_message(coord, shape, vertex)
        elif self._mode == Mode.VERTEX_REMOVE:
            shape, vertex = self.get_value(coord)
            if vertex is not None:
                # have clicked on a current vertex so remove
                index = shape
                shape_type = type(self._data_view.shapes[index])
                if shape_type == Ellipse:
                    # Removing vertex from ellipse not implemented
                    return
                vertices = self._data_view._vertices[
                    self._data_view._index == index
                ]
                if len(vertices) <= 2:
                    # If only 2 vertices present, remove whole shape
                    with self.freeze_refresh():
                        if index in self.selected_data:
                            self.selected_data.remove(index)
                        self._data_view.remove(index)
                        shapes = self.selected_data
                        self._selected_box = self.interaction_box(shapes)
                elif shape_type == Polygon and len(vertices) == 3:
                    # If only 3 vertices of a polygon present remove
                    with self.freeze_refresh():
                        if index in self.selected_data:
                            self.selected_data.remove(index)
                        self._data_view.remove(index)
                        shapes = self.selected_data
                        self._selected_box = self.interaction_box(shapes)
                else:
                    if shape_type == Rectangle:
                        # Deleting vertex from a rectangle creates a polygon
                        new_type = Polygon
                    else:
                        new_type = None
                    # Remove clicked on vertex
                    vertices = np.delete(vertices, vertex, axis=0)
                    with self.freeze_refresh():
                        self._data_view.edit(
                            index, vertices, new_type=new_type
                        )
                        shapes = self.selected_data
                        self._selected_box = self.interaction_box(shapes)
                shape, vertex = self.get_value(coord)
                self._hover_shape = shape
                self._hover_vertex = vertex
                self.refresh()
                self.status = self.get_message(coord, shape, vertex)
        else:
            raise ValueError("Mode not recongnized")

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if event.pos is None:
            return
        self.position = tuple(event.pos)
        coord = self.coordinates[-2:]

        if self._mode == Mode.PAN_ZOOM:
            # If in pan/zoom mode just look at coord all
            shape, vertex = self.get_value(coord)
        elif self._mode == Mode.SELECT:
            if event.is_dragging:
                # Drag any selected shapes
                self._move(coord)
            elif self._is_moving:
                pass
            elif self._is_selecting:
                pass
            else:
                # Highlight boxes if hover over any
                self._hover_shape, self._hover_vertex = self.get_value(coord)
                self._set_highlight()
            shape = self._hover_shape
            vertex = self._hover_vertex
        elif self._mode == Mode.DIRECT:
            if event.is_dragging:
                # Drag any selected shapes
                self._move(coord)
            elif self._is_moving:
                pass
            elif self._is_selecting:
                pass
            else:
                # Highlight boxes if hover over any
                self._hover_shape, self._hover_vertex = self.get_value(coord)
                self._set_highlight()
            shape = self._hover_shape
            vertex = self._hover_vertex
        elif self._mode in (
            [Mode.ADD_RECTANGLE, Mode.ADD_ELLIPSE, Mode.ADD_LINE]
        ):
            # While drawing a shape or doing nothing
            if self._is_creating and event.is_dragging:
                # Drag any selected shapes
                self._move(coord)
                shape = self._hover_shape
                vertex = self._hover_vertex
            else:
                shape, vertex = self.get_value(coord)
        elif self._mode in [Mode.ADD_PATH, Mode.ADD_POLYGON]:
            # While drawing a path or doing nothing
            if self._is_creating:
                # Drag any selected shapes
                self._move(coord)
                shape = self._hover_shape
                vertex = self._hover_vertex
            else:
                shape, vertex = self.get_value(coord)
        elif self._mode in [Mode.VERTEX_INSERT, Mode.VERTEX_REMOVE]:
            self._hover_shape, self._hover_vertex = self.get_value(coord)
            self._set_highlight()
            shape = self._hover_shape
            vertex = self._hover_vertex
        else:
            raise ValueError("Mode not recongnized")

        self.status = self.get_message(coord, shape, vertex)

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if event.pos is None:
            return
        self.position = tuple(event.pos)
        coord = self.coordinates[-2:]
        shift = 'Shift' in event.modifiers

        if self._mode == Mode.PAN_ZOOM:
            # If in pan/zoom mode do nothing
            pass
        elif self._mode == Mode.SELECT:
            shape, vertex = self.get_value(coord)
            if not self._is_moving and not self._is_selecting and not shift:
                if shape is not None:
                    self.selected_data = [shape]
                else:
                    self.selected_data = []
            elif self._is_selecting:
                self.selected_data = self._data_view.shapes_in_box(
                    self._drag_box
                )
                self._is_selecting = False
                self._set_highlight()
            self._is_moving = False
            self._drag_start = None
            self._drag_box = None
            self._fixed_vertex = None
            self._moving_shape = None
            self._moving_vertex = None
            self._hover_shape = shape
            self._hover_vertex = shape
            self._set_highlight()
            self.status = self.get_message(coord, shape, vertex)
            self._update_thumbnail()
        elif self._mode == Mode.DIRECT:
            shape, vertex = self.get_value(coord)
            if not self._is_moving and not self._is_selecting and not shift:
                if shape is not None:
                    self.selected_data = [shape]
                else:
                    self.selected_data = []
            elif self._is_selecting:
                self.selected_data = self._data_view.shapes_in_box(
                    self._drag_box
                )
                self._is_selecting = False
                self._set_highlight()
            self._is_moving = False
            self._drag_start = None
            self._drag_box = None
            self._fixed_vertex = None
            self._moving_shape = None
            self._moving_vertex = None
            self._hover_shape = shape
            self._hover_vertex = shape
            self._set_highlight()
            self.status = self.get_message(coord, shape, vertex)
            self._update_thumbnail()
        elif self._mode in (
            [Mode.ADD_RECTANGLE, Mode.ADD_ELLIPSE, Mode.ADD_LINE]
        ):
            self._finish_drawing()
            shape, vertex = self.get_value(coord)
            self.status = self.get_message(coord, shape, vertex)
        elif self._mode in (
            [
                Mode.ADD_PATH,
                Mode.ADD_POLYGON,
                Mode.VERTEX_INSERT,
                Mode.VERTEX_REMOVE,
            ]
        ):
            pass
        else:
            raise ValueError("Mode not recongnized")

    def on_key_press(self, event):
        """Called whenever key pressed in canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if event.native.isAutoRepeat():
            return
        else:
            if event.key == ' ':
                if self._mode != Mode.PAN_ZOOM:
                    self._mode_history = self._mode
                    self._selected_data_history = copy(self.selected_data)
                    self.mode = Mode.PAN_ZOOM
                else:
                    self._mode_history = Mode.PAN_ZOOM
            elif event.key == 'Shift':
                self._fixed_aspect = True
                box = self._selected_box
                if box is not None:
                    size = box[Box.BOTTOM_RIGHT] - box[Box.TOP_LEFT]
                    if not np.any(size == np.zeros(2)):
                        self._aspect_ratio = abs(size[1] / size[0])
                    else:
                        self._aspect_ratio = 1
                else:
                    self._aspect_ratio = 1
                if self._is_moving:
                    self._move(self.coordinates[-2:])
            elif event.key == 'r':
                self.mode = Mode.ADD_RECTANGLE
            elif event.key == 'e':
                self.mode = Mode.ADD_ELLIPSE
            elif event.key == 'l':
                self.mode = Mode.ADD_LINE
            elif event.key == 't':
                self.mode = Mode.ADD_PATH
            elif event.key == 'p':
                self.mode = Mode.ADD_POLYGON
            elif event.key == 'd':
                self.mode = Mode.DIRECT
            elif event.key == 's':
                self.mode = Mode.SELECT
            elif event.key == 'z':
                self.mode = Mode.PAN_ZOOM
            elif event.key == 'i':
                self.mode = Mode.VERTEX_INSERT
            elif event.key == 'x':
                self.mode = Mode.VERTEX_REMOVE
            elif event.key == 'c' and 'Control' in event.modifiers:
                if self._mode in [Mode.DIRECT, Mode.SELECT]:
                    self._copy_data()
            elif event.key == 'v' and 'Control' in event.modifiers:
                if self._mode in [Mode.DIRECT, Mode.SELECT]:
                    self._paste_data()
            elif event.key == 'a':
                if self._mode in [Mode.DIRECT, Mode.SELECT]:
                    self.selected_data = list(range(self._nshapes_view))
                    self._set_highlight()
            elif event.key == 'Backspace':
                self.remove_selected()
            elif event.key == 'Escape':
                self._finish_drawing()

    def on_key_release(self, event):
        """Called whenever key released in canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if event.key == ' ':
            if self._mode_history != Mode.PAN_ZOOM:
                self.mode = self._mode_history
                self.selected_data = self._selected_data_history
                self._set_highlight()
        elif event.key == 'Shift':
            self._fixed_aspect = False
            if self._is_moving:
                self._move(self.coordinates[-2:])
