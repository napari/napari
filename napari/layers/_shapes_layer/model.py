import numpy as np
from copy import copy, deepcopy
from contextlib import contextmanager

from ...util.event import Event
from ...util.misc import ensure_iterable
from .._base_layer import Layer
from .._register import add_to_viewer

from ..._vispy.scene.visuals import Mesh, Markers, Compound
from ..._vispy.scene.visuals import Line as VispyLine
from vispy.color import get_color_names

from .view import QtShapesLayer
from .view import QtShapesControls
from ._constants import (Mode, BOX_LINE_HANDLE, BOX_LINE, BOX_TOP_CENTER,
                         BOX_CENTER, BOX_LEN, BOX_HANDLE, BOX_WITH_HANDLE,
                         BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, BOX_BOTTOM_LEFT,
                         BACKSPACE)
from .shape_list import ShapeList
from .shape_util import create_box, point_to_lines
from .shapes import Rectangle, Ellipse, Line, Path, Polygon


@add_to_viewer
class Shapes(Layer):
    """Shapes layer.

    Parameters
    ----------
    data : np.array | list
        List of np.array of data or np.array. Each element of the list
        (or row of a 3D np.array) corresponds to one shape. If a 2D array is
        passed it corresponds to just a single shape.
    shape_type : string | list
        String of shape shape_type, must be one of "{'line', 'rectangle',
        'ellipse', 'path', 'polygon'}". If a list is supplied it must be the
        same length as the length of `data` and each element will be applied to
        each shape otherwise the same value will be used for all shapes.
    edge_width : float | list
        thickness of lines and edges. If a list is supplied it must be the same
        length as the length of `data` and each element will be applied to each
        shape otherwise the same value will be used for all shapes.
    edge_color : str | tuple | list
        If string can be any color name recognized by vispy or hex value if
        starting with `#`. If array-like must be 1-dimensional array with 3 or
        4 elements. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape
        otherwise the same value will be used for all shapes.
    face_color : str | tuple | list
        If string can be any color name recognized by vispy or hex value if
        starting with `#`. If array-like must be 1-dimensional array with 3 or
        4 elements. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape
        otherwise the same value will be used for all shapes.
    opacity : float | list
        Opacity of the shapes, must be between 0 and 1.
    z_index : int | list
        Specifier of z order priority. Shapes with higher z order are displayed
        ontop of others. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape
        otherwise the same value will be used for all shapes.
    name : str, keyword-only
        Name of the layer.

    Attributes
    ----------
    data : ShapeList
        Object containing all the shape data.
    edge_width : float
        thickness of lines and edges.
    edge_color : str
        Color of the shape edge.
    face_color : str
        Color of the shape face.
    opacity : float
        Opacity value between 0.0 and 1.0.
    selected_shapes : list
        List of currently selected shapes.
    mode : Mode
        Interactive mode.

    Extended Summary
    ----------
    _mode_history : Mode
        Interactive mode captured on press of <space>.
    _selected_shapes_history : list
        List of currently selected captured on press of <space>.
    _selected_shapes_stored : list
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
    _cursor_coord : np.ndarray
        Length 2 array of the current cursor position in Image coordinates.
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
    """

    _colors = get_color_names()
    _vertex_size = 10
    _rotation_handle_length = 20
    _highlight_color = (0, 0.6, 1)
    _highlight_width = 1.5

    def __init__(self, data, *, shape_type='rectangle', edge_width=1,
                 edge_color='black', face_color='white', opacity=0.7,
                 z_index=0, name=None):

        # Create a compound visual with the following four subvisuals:
        # Markers: corresponding to the vertices of the interaction box or the
        # shapes that are used for highlights.
        # Lines: The lines of the interaction box used for highlights.
        # Mesh: The mesh of the outlines for each shape used for highlights.
        # Mesh: The actual meshes of the shape faces and edges
        visual = Compound([Markers(), VispyLine(), Mesh(), Mesh()])

        super().__init__(visual, name)

        # Freeze refreshes to prevent drawing before the viewer is constructed
        with self.freeze_refresh():
            # Add the shape data
            self.data = ShapeList()
            self.add_shapes(data, shape_type=shape_type, edge_width=edge_width,
                            edge_color=edge_color, face_color=face_color,
                            opacity=opacity, z_index=z_index)

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
            self._opacity = opacity

            # update flags
            self._need_display_update = False
            self._need_visual_update = False

            self._selected_shapes = []
            self._selected_shapes_stored = []
            self._selected_shapes_history = []
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
            self._cursor_coord = np.array([0, 0])
            self._is_creating = False
            self._update_properties = True
            self._clipboard = []

            self._mode = Mode.PAN_ZOOM
            self._mode_history = self._mode
            self._status = str(self._mode)
            self._help = 'enter a selection mode to edit shape properties'

            self.events.add(mode=Event,
                            edge_width=Event,
                            edge_color=Event,
                            face_color=Event)

            self._qt_properties = QtShapesLayer(self)
            self._qt_controls = QtShapesControls(self)

            self.events.deselect.connect(lambda x: self._finish_drawing())

    @property
    def data(self):
        """ShapeList: object containing all the shape data
        """
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.refresh()

    @property
    def edge_width(self):
        """float: width of edges in px
        """
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width):
        self._edge_width = edge_width
        if self._update_properties:
            index = self.selected_shapes
            for i in index:
                self.data.update_edge_width(i, edge_width)
            self.refresh()
        self.events.edge_width()

    @property
    def edge_color(self):
        """str: color of edges and lines
        """
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color):
        self._edge_color = edge_color
        if self._update_properties:
            index = self.selected_shapes
            for i in index:
                self.data.update_edge_color(i, edge_color)
            self.refresh()
        self.events.edge_color()

    @property
    def face_color(self):
        """str: color of faces
        """
        return self._face_color

    @face_color.setter
    def face_color(self, face_color):
        self._face_color = face_color
        if self._update_properties:
            index = self.selected_shapes
            for i in index:
                self.data.update_face_color(i, face_color)
            self.refresh()
        self.events.face_color()

    @property
    def opacity(self):
        """float: Opacity value between 0.0 and 1.0.
        """
        return self._opacity

    @opacity.setter
    def opacity(self, opacity):
        if not 0.0 <= opacity <= 1.0:
            raise ValueError('opacity must be between 0.0 and 1.0; '
                             f'got {opacity}')

        self._opacity = opacity
        if self._update_properties:
            index = self.selected_shapes
            for i in index:
                self.data.update_opacity(i, opacity)
            self.refresh()
        self.events.opacity()

    @property
    def selected_shapes(self):
        """list: list of currently selected shapes
        """
        return self._selected_shapes

    @selected_shapes.setter
    def selected_shapes(self, selected_shapes):
        self._selected_shapes = selected_shapes
        self._selected_box = self.interaction_box(selected_shapes)

        # Update properties based on selected shapes
        face_colors = list(set([self.data.shapes[i]._face_color_name for i in
                           selected_shapes]))
        if len(face_colors) == 1:
            face_color = face_colors[0]
            with self.block_update_properties():
                self.face_color = face_color

        edge_colors = list(set([self.data.shapes[i]._edge_color_name for i in
                           selected_shapes]))
        if len(edge_colors) == 1:
            edge_color = edge_colors[0]
            with self.block_update_properties():
                self.edge_color = edge_color

        edge_width = list(set([self.data.shapes[i].edge_width for i in
                          selected_shapes]))
        if len(edge_width) == 1:
            edge_width = edge_width[0]
            with self.block_update_properties():
                self.edge_width = edge_width

        opacities = list(set([self.data.shapes[i].opacity for i in
                         selected_shapes]))
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
        return self._mode

    @mode.setter
    def mode(self, mode):
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
            self.help = ('hold <space> to pan/zoom, '
                         f'press <{BACKSPACE}> to remove selected')
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
            self.help = ('hold <space> to pan/zoom, '
                         'press <esc> to finish drawing')
        else:
            raise ValueError("Mode not recongnized")

        self.status = str(mode)
        self._mode = mode

        draw_modes = ([Mode.SELECT, Mode.DIRECT, Mode.VERTEX_INSERT,
                      Mode.VERTEX_REMOVE])

        self.events.mode(mode=mode)
        if not (mode in draw_modes and old_mode in draw_modes):
            self._finish_drawing()
        self.refresh()

    @contextmanager
    def block_update_properties(self):
        self._update_properties = False
        yield
        self._update_properties = True

    def _get_shape(self):
        """Determines the shape of the vertex data.
        """
        if len(self.data._vertices) == 0:
            return [1, 1]
        else:
            return np.max(self.data._vertices, axis=0) + 1

    def add_shapes(self, data, *, shape_type='rectangle', edge_width=1,
                   edge_color='black', face_color='white', opacity=0.7,
                   z_index=0):
        """Add shapes to the current layer.

        Parameters
        ----------
        data : np.array | list
            List of np.array of data or np.array. Each element of the list
            (or row of a 3D np.array) corresponds to one shape. If a 2D array
            is passed it corresponds to just a single shape.
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
        if len(data) == 0:
            return

        if np.array(data[0]).ndim == 1:
            # If a single array for a shape has been passed
            if shape_type in self.data._types.keys():
                shape_cls = self.data._types[shape_type]
                shape = shape_cls(data, edge_width=edge_width,
                                  edge_color=edge_color, face_color=face_color,
                                  opacity=opacity, z_index=z_index)
            else:
                raise ValueError("""shape_type not recognized, must be one of
                                 "{'line', 'rectangle', 'ellipse', 'path',
                                 'polygon'}"
                                 """)
            self.data.add(shape)
        else:
            # Turn input arguments into iterables
            shape_types = ensure_iterable(shape_type)
            edge_widths = ensure_iterable(edge_width)
            opacities = ensure_iterable(opacity)
            z_indices = ensure_iterable(z_index)
            edge_colors = ensure_iterable(edge_color, color=True)
            face_colors = ensure_iterable(face_color, color=True)

            for d, st, ew, ec, fc, o, z, in zip(data, shape_types, edge_widths,
                                                edge_colors, face_colors,
                                                opacities, z_indices):
                shape_cls = self.data._types[st]
                shape = shape_cls(d, edge_width=ew, edge_color=ec,
                                  face_color=fc, opacity=o, z_index=z)
                self.data.add(shape)

    def _update(self):
        """Update the underlying visual.
        """
        if self._need_display_update:
            self._need_display_update = False
            self._set_view_slice()

        if self._need_visual_update:
            self._need_visual_update = False
            self._node.update()

    def _refresh(self):
        """Fully refresh the underlying visual.
        """
        self._need_display_update = True
        self._update()

    def _set_view_slice(self, indices=None):
        """Set the shape mesh data to the view.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.
        """
        z_order = self.data._mesh.triangles_z_order
        faces = self.data._mesh.triangles[z_order]
        colors = self.data._mesh.triangles_colors[z_order]
        vertices = self.data._mesh.vertices
        if len(faces) == 0:
            self._node._subvisuals[3].set_data(vertices=None, faces=None)
        else:
            self._node._subvisuals[3].set_data(vertices=vertices, faces=faces,
                                               face_colors=colors)
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
                box = copy(self.data.shapes[index[0]]._box)
            else:
                indices = np.isin(self.data._index, index)
                box = create_box(self.data._vertices[indices])
        else:
            box = copy(self.data.shapes[index]._box)

        if box is not None:
            rot = box[BOX_TOP_CENTER]
            length_box = np.linalg.norm(box[BOX_BOTTOM_LEFT] -
                                        box[BOX_TOP_LEFT])
            if length_box > 0:
                rescale = self._get_rescale()
                r = self._rotation_handle_length*rescale
                rot = rot-r*(box[BOX_BOTTOM_LEFT] -
                             box[BOX_TOP_LEFT])/length_box
            box = np.append(box, [rot], axis=0)

        return box

    def _outline_shapes(self):
        """Finds outlines of any selected shapes including any shape hovered
        over

        Returns
        ----------
        vertices : None | np.ndarray
            Nx2 array of any vertices of outline or None
        triangles : None | np.ndarray
            Mx3 array of any indices of vertices for triangles of outline or
            None
        """
        if self._hover_shape is not None or len(self.selected_shapes) > 0:
            if len(self.selected_shapes) > 0:
                index = copy(self.selected_shapes)
                if self._hover_shape is not None:
                    if self._hover_shape in index:
                        pass
                    else:
                        index.append(self._hover_shape)
                index.sort()
            else:
                index = self._hover_shape

            centers, offsets, triangles = self.data.outline(index)
            rescale = self._get_rescale()
            vertices = centers + rescale*self._highlight_width*offsets
        else:
            vertices = None
            triangles = None

        return vertices, triangles

    def _compute_vertices_and_box(self):
        """Compute the location and properties of the vertices and box that
        need to get rendered

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
        if len(self.selected_shapes) > 0:
            if self.mode == Mode.SELECT:
                # If in select mode just show the interaction boudning box
                # including its vertices and the rotation handle
                box = self._selected_box[BOX_WITH_HANDLE]
                if self._hover_shape is None:
                    face_color = 'white'
                elif self._hover_vertex is None:
                    face_color = 'white'
                else:
                    face_color = self._highlight_color
                edge_color = self._highlight_color
                vertices = box
                # Use a subset of the vertices of the interaction_box to plot
                # the line around the edge
                pos = box[BOX_LINE_HANDLE]
                width = 1.5
            elif self.mode in ([Mode.DIRECT, Mode.ADD_PATH, Mode.ADD_POLYGON,
                                Mode.ADD_RECTANGLE, Mode.ADD_ELLIPSE,
                                Mode.ADD_LINE, Mode.VERTEX_INSERT,
                                Mode.VERTEX_REMOVE]):
                # If in one of these mode show the vertices of the shape itself
                inds = np.isin(self.data._index, self.selected_shapes)
                vertices = self.data._vertices[inds]
                # If currently adding path don't show box over last vertex
                if self.mode == Mode.ADD_PATH:
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
            pos = box[BOX_LINE]
        else:
            # Otherwise show nothing
            vertices = np.empty((0, 2))
            face_color = 'white'
            edge_color = 'white'
            pos = None
            width = 0

        return vertices, face_color, edge_color, pos, width

    def _set_highlight(self, force=False):
        """Render highlights of shapes including boundaries, vertices,
        interaction boxes, and the drag selection box when appropriate

        Parameters
        ----------
        force : bool
            Bool that forces a redraw to occur when `True`
        """
        # Check if any shape or vertex ids have changed since last call
        if (self.selected_shapes == self._selected_shapes_stored and
                self._hover_shape == self._hover_shape_stored and
                self._hover_vertex == self._hover_vertex_stored and
                np.all(self._drag_box == self._drag_box_stored)) and not force:
            return
        self._selected_shapes_stored = copy(self.selected_shapes)
        self._hover_shape_stored = copy(self._hover_shape)
        self._hover_vertex_stored = copy(self._hover_vertex)
        self._drag_box_stored = copy(self._drag_box)

        # Compute the vertices and faces of any shape outlines
        vertices, faces = self._outline_shapes()
        self._node._subvisuals[2].set_data(vertices=vertices, faces=faces,
                                           color=self._highlight_color)

        # Compute the location and properties of the vertices and box that
        # need to get rendered
        (vertices, face_color, edge_color, pos,
            width) = self._compute_vertices_and_box()
        self._node._subvisuals[0].set_data(vertices, size=self._vertex_size,
                                           face_color=face_color,
                                           edge_color=edge_color,
                                           edge_width=1.5, symbol='square',
                                           scaling=False)
        self._node._subvisuals[1].set_data(pos=pos, color=edge_color,
                                           width=width)

    def _finish_drawing(self):
        """Reset properties used in shape drawing so new shapes can be drawn.
        """
        index = copy(self._moving_shape)
        self._is_moving = False
        self.selected_shapes = []
        self._drag_start = None
        self._drag_box = None
        self._fixed_vertex = None
        self._moving_shape = None
        self._moving_vertex = None
        self._hover_shape = None
        self._hover_vertex = None
        if self._is_creating is True and self.mode == Mode.ADD_PATH:
            vertices = self.data._vertices[self.data._index == index]
            if len(vertices) <= 2:
                self.data.remove(index)
            else:
                self.data.edit(index, vertices[:-1])
        if self._is_creating is True and self.mode == Mode.ADD_POLYGON:
            vertices = self.data._vertices[self.data._index == index]
            if len(vertices) <= 2:
                self.data.remove(index)
        self._is_creating = False
        self.refresh()

    def remove_selected(self):
        """Remove any selected shapes.
        """
        to_remove = sorted(self.selected_shapes, reverse=True)
        for index in to_remove:
            self.data.remove(index)
        self.selected_shapes = []
        shape, vertex = self._shape_at(self._cursor_coord)
        self._hover_shape = shape
        self._hover_vertex = vertex
        self.status = self.get_message(self._cursor_coord, shape, vertex)
        self.refresh()

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
        transform = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
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
        box = np.array(box*scale)
        if not np.all(box[BOX_TOP_CENTER] == box[BOX_HANDLE]):
            rescale = self._get_rescale()
            r = self._rotation_handle_length*rescale
            handle_vec = box[BOX_HANDLE]-box[BOX_TOP_CENTER]
            cur_len = np.linalg.norm(handle_vec)
            box[BOX_HANDLE] = box[BOX_TOP_CENTER] + r*handle_vec/cur_len
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
        if not np.all(box[BOX_TOP_CENTER] == box[BOX_HANDLE]):
            rescale = self._get_rescale()
            r = self._rotation_handle_length*rescale
            handle_vec = box[BOX_HANDLE]-box[BOX_TOP_CENTER]
            cur_len = np.linalg.norm(handle_vec)
            box[BOX_HANDLE] = box[BOX_TOP_CENTER] + r*handle_vec/cur_len
        self._selected_box = box + center

    def _shape_at(self, coord):
        """Determines if any shape at given coord by looking inside triangle
        meshes.

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
        if len(self.selected_shapes) > 0:
            if self.mode == Mode.SELECT:
                # Check if inside vertex of interaction box or rotation handle
                box = self._selected_box[BOX_WITH_HANDLE]
                distances = abs(box - coord[:2])

                # Get the vertex sizes
                rescale = self._get_rescale()
                sizes = self._vertex_size*rescale/2

                # Check if any matching vertices
                matches = np.all(distances <= sizes, axis=1).nonzero()
                if len(matches[0]) > 0:
                    return self.selected_shapes[0], matches[0][-1]
            elif self.mode in ([Mode.DIRECT, Mode.VERTEX_INSERT,
                                Mode.VERTEX_REMOVE]):
                # Check if inside vertex of shape
                inds = np.isin(self.data._index, self.selected_shapes)
                vertices = self.data._vertices[inds]
                distances = abs(vertices - coord[:2])

                # Get the vertex sizes
                rescale = self._get_rescale()
                sizes = self._vertex_size*rescale/2

                # Check if any matching vertices
                matches = np.all(distances <= sizes, axis=1).nonzero()[0]
                if len(matches) > 0:
                    index = inds.nonzero()[0][matches[-1]]
                    shape = self.data._index[index]
                    _, idx = np.unique(self.data._index, return_index=True)
                    return shape, index - idx[shape]

        # Check if mouse inside shape
        shape = self.data.inside(coord)
        return shape, None

    def _get_rescale(self):
        """Get conversion factor from canvas coordinates to image coordinates.
        Depends on the current zoom level.

        Returns
        ----------
        rescale : float
            Conversion factor from canvas coordinates to image coordinates.
        """
        transform = self.viewer._canvas.scene.node_transform(self._node)
        rescale = transform.map([1, 1])[:2] - transform.map([0, 0])[:2]

        return rescale.mean()

    def _get_coord(self, position):
        """Convert a position in canvas coordinates to image coordinates.

        Parameters
        ----------
        position : sequence of int
            Position of mouse cursor in canvas coordinates.

        Returns
        ----------
        coord : sequence of float
            Position of mouse cursor in image coordinates.
        """
        transform = self.viewer._canvas.scene.node_transform(self._node)
        pos = transform.map(position)
        coord = np.array([pos[0], pos[1]])
        self._cursor_coord = coord

        return coord

    def get_message(self, coord, shape, vertex):
        """Generates a string based on the coordinates and information about
        what shapes are hovered over

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
        coord_shift = copy(coord)
        coord_shift[0] = int(coord[1])
        coord_shift[1] = int(coord[0])
        msg = f'{coord_shift.astype(int)}, {self.name}'
        if shape is not None:
            msg = msg + ', shape ' + str(shape)
            if vertex is not None:
                msg = msg + ', vertex ' + str(vertex)
        return msg

    def move_to_front(self):
        """Moves selected objects to be displayed in front of all others.
        """
        if len(self.selected_shapes) == 0:
            return
        new_z_index = max(self.data._z_index) + 1
        for index in self.selected_shapes:
            self.data.update_z_index(index, new_z_index)
        self.refresh()

    def move_to_back(self):
        """Moves selected objects to be displayed behind all others.
        """
        if len(self.selected_shapes) == 0:
            return
        new_z_index = min(self.data._z_index) - 1
        for index in self.selected_shapes:
            self.data.update_z_index(index, new_z_index)
        self.refresh()

    def _copy_shapes(self):
        """Copy selected shapes to clipboard.
        """
        self._clipboard = ([deepcopy(self.data.shapes[i]) for i in
                           self._selected_shapes])

    def _paste_shapes(self):
        """Paste any shapes from clipboard and then selects them.
        """
        cur_shapes = len(self.data.shapes)
        for s in self._clipboard:
            self.data.add(s)
        self.selected_shapes = list(range(cur_shapes,
                                    cur_shapes+len(self._clipboard)))
        self.move_to_front()
        self._copy_shapes()

    def _move(self, coord):
        """Moves object at given mouse position and set of indices.

        Parameters
        ----------
        coord : sequence of two int
            Position of mouse cursor in image coordinates.
        """
        vertex = self._moving_vertex
        if self.mode in ([Mode.SELECT, Mode.ADD_RECTANGLE, Mode.ADD_ELLIPSE,
                         Mode.ADD_LINE]):
            if len(self.selected_shapes) > 0:
                self._is_moving = True
                if vertex is None:
                    # Check where dragging box from to move whole object
                    if self._drag_start is None:
                        center = self._selected_box[BOX_CENTER]
                        self._drag_start = coord - center
                    center = self._selected_box[BOX_CENTER]
                    shift = coord - center - self._drag_start
                    for index in self.selected_shapes:
                        self.data.shift(index, shift)
                    self._selected_box = self._selected_box + shift
                    self.refresh()
                elif vertex < BOX_LEN:
                    # Corner / edge vertex is being dragged so resize object
                    box = self._selected_box
                    if self._fixed_vertex is None:
                        self._fixed_index = (vertex+4) % BOX_LEN
                        self._fixed_vertex = box[self._fixed_index]

                    size = (box[(self._fixed_index+4) % BOX_LEN] -
                            box[self._fixed_index])
                    offset = box[BOX_HANDLE] - box[BOX_CENTER]
                    offset = offset/np.linalg.norm(offset)
                    offset_perp = np.array([offset[1], -offset[0]])

                    fixed = self._fixed_vertex
                    new = copy(coord)

                    if self._fixed_aspect and self._fixed_index % 2 == 0:
                        if (new - fixed)[0] == 0:
                            ratio = 1
                        else:
                            ratio = abs((new - fixed)[1]/(new - fixed)[0])
                        if ratio > self._aspect_ratio:
                            r = self._aspect_ratio/ratio
                            new[1] = fixed[1]+(new[1]-fixed[1])*r
                        else:
                            r = ratio/self._aspect_ratio
                            new[0] = fixed[0]+(new[0]-fixed[0])*r

                    if size @ offset == 0:
                        dist = 1
                    else:
                        dist = ((new - fixed) @ offset) / (size @ offset)

                    if size @ offset_perp == 0:
                        dist_perp = 1
                    else:
                        dist_perp = (((new - fixed) @ offset_perp) /
                                     (size @ offset_perp))

                    if self._fixed_index % 2 == 0:
                        # corner selected
                        scale = np.array([dist_perp, dist])
                    elif self._fixed_index % 4 == 1:
                        # top selected
                        scale = np.array([1, dist])
                    else:
                        # side selected
                        scale = np.array([dist_perp, 1])

                    # prevent box from shrinking below a threshold size
                    rescale = self._get_rescale()
                    threshold = self._vertex_size*rescale/8
                    scale[abs(scale*size) < threshold] = 1

                    # check orientation of box
                    angle = -np.arctan2(offset[0], -offset[1])
                    c, s = np.cos(angle), np.sin(angle)
                    if angle == 0:
                        for index in self.selected_shapes:
                            self.data.scale(index, scale,
                                            center=self._fixed_vertex)
                        self._scale_box(scale, center=self._fixed_vertex)
                    else:
                        rotation = np.array([[c, s], [-s, c]])
                        scale_mat = np.array([[scale[0], 0], [0, scale[1]]])
                        inv_rot = np.array([[c, -s], [s, c]])
                        transform = rotation @ scale_mat @ inv_rot
                        for index in self.selected_shapes:
                            self.data.shift(index, -self._fixed_vertex)
                            self.data.transform(index, transform)
                            self.data.shift(index, self._fixed_vertex)
                        self._transform_box(transform,
                                            center=self._fixed_vertex)
                    self.refresh()
                elif vertex == 8:
                    # Rotation handle is being dragged so rotate object
                    handle = self._selected_box[BOX_HANDLE]
                    if self._drag_start is None:
                        self._fixed_vertex = self._selected_box[BOX_CENTER]
                        offset = handle - self._fixed_vertex
                        self._drag_start = -np.degrees(np.arctan2(offset[0],
                                                       -offset[1]))

                    new_offset = coord - self._fixed_vertex
                    new_angle = -np.degrees(np.arctan2(new_offset[0],
                                            -new_offset[1]))
                    fixed_offset = handle - self._fixed_vertex
                    fixed_angle = -np.degrees(np.arctan2(fixed_offset[0],
                                              -fixed_offset[1]))

                    if np.linalg.norm(new_offset) < 1:
                        angle = 0
                    elif self._fixed_aspect:
                        angle = np.round(new_angle/45)*45 - fixed_angle
                    else:
                        angle = new_angle - fixed_angle

                    for index in self.selected_shapes:
                        self.data.rotate(index, angle,
                                         center=self._fixed_vertex)
                    self._rotate_box(angle, center=self._fixed_vertex)
                    self.refresh()
            else:
                self._is_selecting = True
                if self._drag_start is None:
                    self._drag_start = coord
                self._drag_box = np.array([self._drag_start, coord])
                self._set_highlight()
        elif self.mode in [Mode.DIRECT, Mode.ADD_PATH, Mode.ADD_POLYGON]:
            if len(self.selected_shapes) > 0:
                if vertex is not None:
                    self._is_moving = True
                    index = self._moving_shape
                    shape_type = type(self.data.shapes[index])
                    if shape_type == Ellipse:
                        # DIRECT vertex moving of ellipse not implemented
                        pass
                    else:
                        if shape_type == Rectangle:
                            new_type = Polygon
                        else:
                            new_type = None
                        indices = self.data._index == index
                        vertices = self.data._vertices[indices]
                        vertices[vertex] = coord
                        self.data.edit(index, vertices, new_type=new_type)
                        shapes = self.selected_shapes
                        self._selected_box = self.interaction_box(shapes)
                        self.refresh()
            else:
                self._is_selecting = True
                if self._drag_start is None:
                    self._drag_start = coord
                self._drag_box = np.array([self._drag_start, coord])
                self._set_highlight()
        elif self.mode in [Mode.VERTEX_INSERT, Mode.VERTEX_REMOVE]:
            if len(self.selected_shapes) > 0:
                pass
            else:
                self._is_selecting = True
                if self._drag_start is None:
                    self._drag_start = coord
                self._drag_box = np.array([self._drag_start, coord])
                self._set_highlight()

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        position = event.pos
        coord = self._get_coord(position)
        shift = 'Shift' in event.modifiers

        if self.mode == Mode.PAN_ZOOM:
            # If in pan/zoom mode do nothing
            pass
        elif self.mode in [Mode.SELECT, Mode.DIRECT]:
            if not self._is_moving and not self._is_selecting:
                shape, vertex = self._shape_at(coord)
                self._moving_shape = shape
                self._moving_vertex = vertex
                if vertex is None:
                    if shift and shape is not None:
                        if shape in self.selected_shapes:
                            self.selected_shapes.remove(shape)
                            shapes = self.selected_shapes
                            self._selected_box = self.interaction_box(shapes)
                        else:
                            self.selected_shapes.append(shape)
                            shapes = self.selected_shapes
                            self._selected_box = self.interaction_box(shapes)
                    elif shape is not None:
                        if shape not in self.selected_shapes:
                            self.selected_shapes = [shape]
                    else:
                        self.selected_shapes = []
                self._set_highlight()
                self.status = self.get_message(coord, shape, vertex)
        elif self.mode in ([Mode.ADD_RECTANGLE, Mode.ADD_ELLIPSE,
                            Mode.ADD_LINE]):
            # Start drawing a rectangle / ellipse / line
            rescale = self._get_rescale()
            size = self._vertex_size*rescale/4
            new_z_index = max(self.data._z_index, default=-1) + 1
            if self.mode == Mode.ADD_RECTANGLE:
                data = np.array([coord, coord+size])
                shape_type = 'rectangle'
            elif self.mode == Mode.ADD_ELLIPSE:
                data = np.array([coord+size/2, [size, size]])
                shape_type = 'ellipse'
            elif self.mode == Mode.ADD_LINE:
                data = np.array([coord, coord+size])
                shape_type = 'line'
            self.add_shapes(data, shape_type=shape_type,
                            edge_width=self.edge_width,
                            edge_color=self.edge_color,
                            face_color=self.face_color,
                            opacity=self.opacity, z_index=new_z_index)
            self.selected_shapes = [len(self.data.shapes)-1]
            ind = 4
            self._moving_shape = self.selected_shapes[0]
            self._moving_vertex = ind
            self._hover_shape = self.selected_shapes[0]
            self._hover_vertex = ind
            self._is_creating = True
            self._set_highlight()
            self.refresh()
        elif self.mode in [Mode.ADD_PATH, Mode.ADD_POLYGON]:
            if self._is_creating is False:
                # Start drawing a path
                data = np.array([coord, coord])
                new_z_index = max(self.data._z_index, default=-1) + 1
                self.add_shapes(data, shape_type='path',
                                edge_width=self.edge_width,
                                edge_color=self.edge_color,
                                face_color=self.face_color,
                                opacity=self.opacity,
                                z_index=new_z_index)
                self.selected_shapes = [len(self.data.shapes)-1]
                ind = 1
                self._moving_shape = self.selected_shapes[0]
                self._moving_vertex = ind
                self._hover_shape = self.selected_shapes[0]
                self._hover_vertex = ind
                self._is_creating = True
                self._set_highlight()
            else:
                # Add to an existing path or polygon
                index = self._moving_shape
                if self.mode == Mode.ADD_POLYGON:
                    new_type = Polygon
                else:
                    new_type = None
                vertices = self.data._vertices[self.data._index == index]
                vertices = np.concatenate((vertices, [coord]),  axis=0)
                # Change the selected vertex
                self._moving_vertex = self._moving_vertex + 1
                self._hover_vertex = self._hover_vertex + 1
                self.data.edit(index, vertices, new_type=new_type)
                self._selected_box = self.interaction_box(self.selected_shapes)
            self.status = self.get_message(coord, self._hover_shape,
                                           self._hover_vertex)
        elif self.mode == Mode.VERTEX_INSERT:
            if len(self.selected_shapes) == 0:
                # If none selected return immediately
                return

            all_lines = np.empty((0, 2, 2))
            all_lines_shape = np.empty((0, 2), dtype=int)
            for index in self.selected_shapes:
                shape_type = type(self.data.shapes[index])
                if shape_type == Ellipse:
                    # Adding vertex to ellipse not implemented
                    pass
                else:
                    vertices = self.data._vertices[self.data._index == index]
                    # Find which edge new vertex should inserted along
                    closed = shape_type != Path
                    n = len(vertices)
                    if closed:
                        lines = np.array([[vertices[i],
                                         vertices[(i+1) % n]] for i in
                                         range(n)])
                    else:
                        lines = np.array([[vertices[i], vertices[i+1]] for i in
                                         range(n-1)])
                    all_lines = np.append(all_lines, lines, axis=0)
                    indices = np.array([np.repeat(index, len(lines)),
                                       list(range(len(lines)))]).T
                    all_lines_shape = np.append(all_lines_shape, indices,
                                                axis=0)
            if len(all_lines) == 0:
                # No appropriate shapes found
                return
            ind, loc = point_to_lines(coord, all_lines)
            index = all_lines_shape[ind][0]
            ind = all_lines_shape[ind][1]+1
            shape_type = type(self.data.shapes[index])
            if shape_type == Line:
                # Adding vertex to line turns it into a path
                new_type = Path
            elif shape_type == Rectangle:
                # Adding vertex to rectangle turns it into a polygon
                new_type = Polygon
            else:
                new_type = None
            closed = shape_type != Path
            vertices = self.data._vertices[self.data._index == index]
            if closed is not True:
                if int(ind) == 1 and loc < 0:
                    ind = 0
                elif int(ind) == len(vertices)-1 and loc > 1:
                    ind = ind + 1

            vertices = np.insert(vertices, ind, [coord], axis=0)
            with self.freeze_refresh():
                self.data.edit(index, vertices, new_type=new_type)
                self._selected_box = self.interaction_box(self.selected_shapes)
            shape, vertex = self._shape_at(coord)
            self._hover_shape = shape
            self._hover_vertex = vertex
            self.refresh()
            self.status = self.get_message(coord, shape, vertex)
        elif self.mode == Mode.VERTEX_REMOVE:
            shape, vertex = self._shape_at(coord)
            if vertex is not None:
                # have clicked on a current vertex so remove
                index = shape
                shape_type = type(self.data.shapes[index])
                if shape_type == Ellipse:
                    # Removing vertex from ellipse not implemented
                    return
                vertices = self.data._vertices[self.data._index == index]
                if len(vertices) <= 2:
                    # If only 2 vertices present, remove whole shape
                    with self.freeze_refresh():
                        if index in self.selected_shapes:
                            self.selected_shapes.remove(index)
                        self.data.remove(index)
                        shapes = self.selected_shapes
                        self._selected_box = self.interaction_box(shapes)
                elif shape_type == Polygon and len(vertices) == 3:
                    # If only 3 vertices of a polygon present remove
                    with self.freeze_refresh():
                        if index in self.selected_shapes:
                            self.selected_shapes.remove(index)
                        self.data.remove(index)
                        shapes = self.selected_shapes
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
                        self.data.edit(index, vertices, new_type=new_type)
                        shapes = self.selected_shapes
                        self._selected_box = self.interaction_box(shapes)
                shape, vertex = self._shape_at(coord)
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
        position = event.pos
        coord = self._get_coord(position)

        if self.mode == Mode.PAN_ZOOM:
            # If in pan/zoom mode just look at coord all
            shape, vertex = self._shape_at(coord)
        elif self.mode == Mode.SELECT:
            if event.is_dragging:
                # Drag any selected shapes
                self._move(coord)
            elif self._is_moving:
                pass
            elif self._is_selecting:
                pass
            else:
                # Highlight boxes if hover over any
                self._hover_shape, self._hover_vertex = self._shape_at(coord)
                self._set_highlight()
            shape = self._hover_shape
            vertex = self._hover_vertex
        elif self.mode == Mode.DIRECT:
            if event.is_dragging:
                # Drag any selected shapes
                self._move(coord)
            elif self._is_moving:
                pass
            elif self._is_selecting:
                pass
            else:
                # Highlight boxes if hover over any
                self._hover_shape, self._hover_vertex = self._shape_at(coord)
                self._set_highlight()
            shape = self._hover_shape
            vertex = self._hover_vertex
        elif self.mode in ([Mode.ADD_RECTANGLE, Mode.ADD_ELLIPSE,
                            Mode.ADD_LINE]):
            # While drawing a shape or doing nothing
            if self._is_creating and event.is_dragging:
                # Drag any selected shapes
                self._move(coord)
                shape = self._hover_shape
                vertex = self._hover_vertex
            else:
                shape, vertex = self._shape_at(coord)
        elif self.mode in [Mode.ADD_PATH, Mode.ADD_POLYGON]:
            # While drawing a path or doing nothing
            if self._is_creating:
                # Drag any selected shapes
                self._move(coord)
                shape = self._hover_shape
                vertex = self._hover_vertex
            else:
                shape, vertex = self._shape_at(coord)
        elif self.mode in [Mode.VERTEX_INSERT, Mode.VERTEX_REMOVE]:
            self._hover_shape, self._hover_vertex = self._shape_at(coord)
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
        position = event.pos
        coord = self._get_coord(position)
        shift = 'Shift' in event.modifiers

        if self.mode == Mode.PAN_ZOOM:
            # If in pan/zoom mode do nothing
            pass
        elif self.mode == Mode.SELECT:
            shape, vertex = self._shape_at(coord)
            if not self._is_moving and not self._is_selecting and not shift:
                if shape is not None:
                    self.selected_shapes = [shape]
                else:
                    self.selected_shapes = []
            elif self._is_selecting:
                self.selected_shapes = self.data.shapes_in_box(self._drag_box)
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
        elif self.mode == Mode.DIRECT:
            shape, vertex = self._shape_at(coord)
            if not self._is_moving and not self._is_selecting and not shift:
                if shape is not None:
                    self.selected_shapes = [shape]
                else:
                    self.selected_shapes = []
            elif self._is_selecting:
                self.selected_shapes = self.data.shapes_in_box(self._drag_box)
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
        elif self.mode in ([Mode.ADD_RECTANGLE, Mode.ADD_ELLIPSE,
                            Mode.ADD_LINE]):
            self._finish_drawing()
            shape, vertex = self._shape_at(coord)
            self.status = self.get_message(coord, shape, vertex)
        elif self.mode in ([Mode.ADD_PATH, Mode.ADD_POLYGON,
                            Mode.VERTEX_INSERT, Mode.VERTEX_REMOVE]):
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
                if self.mode != Mode.PAN_ZOOM:
                    self._mode_history = self.mode
                    self._selected_shapes_history = copy(self.selected_shapes)
                    self.mode = Mode.PAN_ZOOM
                else:
                    self._mode_history = Mode.PAN_ZOOM
            elif event.key == 'Shift':
                self._fixed_aspect = True
                box = self._selected_box
                if box is not None:
                    size = box[BOX_BOTTOM_RIGHT]-box[BOX_TOP_LEFT]
                    if not np.any(size == np.zeros(2)):
                        self._aspect_ratio = abs(size[1] / size[0])
                    else:
                        self._aspect_ratio = 1
                else:
                    self._aspect_ratio = 1
                if self._is_moving:
                    self._move(self._cursor_coord)
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
                if self.mode in [Mode.DIRECT, Mode.SELECT]:
                    self._copy_shapes()
            elif event.key == 'v' and 'Control' in event.modifiers:
                if self.mode in [Mode.DIRECT, Mode.SELECT]:
                    self._paste_shapes()
            elif event.key == 'a':
                if self.mode in [Mode.DIRECT, Mode.SELECT]:
                    self.selected_shapes = list(range(len(self.data.shapes)))
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
                self.selected_shapes = self._selected_shapes_history
                self._set_highlight()
        elif event.key == 'Shift':
            self._fixed_aspect = False
            if self._is_moving:
                self._move(self._cursor_coord)
