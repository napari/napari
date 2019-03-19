import numpy as np
from copy import copy

from ...util.event import Event
from .._base_layer import Layer
from .._register import add_to_viewer

from ..._vispy.scene.visuals import Mesh, Markers, Line, Compound
from vispy.color import get_color_names

from .view import QtShapesLayer
from .view import QtShapesControls
from .shape_list import ShapeList
from .shape_util import create_box, inside_triangles, point_to_lines


@add_to_viewer
class Shapes(Layer):
    """Shapes layer.
    Parameters
    ----------
    data : np.array | list
        List of Shape objects of list of np.array of data or np.array. Each
        element of the list (or now of the np.array) corresponds to one shape.
        If a list of Shape objects is passed the other shape specific keyword
        arguments are ignored.
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
    """

    _colors = get_color_names()
    _vertex_size = 10
    _highlight_color = (0, 0.6, 1)
    _highlight_width = 1.5
    _rotion_handle_length = 20
    _prefixed_size = np.array([10, 10])

    def __init__(self, data, shape_type='rectangle', edge_width=1,
                 edge_color='black', face_color='white', opacity=1, z_index=0,
                 *, name=None):

        visual = Compound([Markers(), Line(), Mesh(), Mesh()])

        super().__init__(visual, name)

        # Freeze refreshes
        with self.freeze_refresh():
            # Add the shape data
            self.data = ShapeList(data, shape_type=shape_type,
                                  edge_width=edge_width,
                                  edge_color=edge_color,
                                  face_color=face_color,
                                  opacity=opacity,
                                  z_index=z_index)

            self._apply_all = True
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
                self._face_color = 'black'
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
            self._mouse_coord = [0, 0]
            self._creating = False

            self._mode = 'pan/zoom'
            self._mode_history = self._mode
            self._status = self._mode

            self.events.add(mode=Event,
                            edge_width=Event,
                            edge_color=Event,
                            face_color=Event,
                            apply_all=Event)

            self._qt_properties = QtShapesLayer(self)
            self._qt_controls = QtShapesControls(self)

            self.events.deselect.connect(lambda x: self._finish_drawing())

    @property
    def data(self):
        """ShapesData: object with shapes data
        """
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.refresh()

    @property
    def edge_width(self):
        """int: width of edges in px
        """
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width):
        self._edge_width = edge_width
        if self._apply_all:
            index = list(range(len(self.data.shapes)))
        else:
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
        if self._apply_all:
            index = list(range(len(self.data.shapes)))
        else:
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
        if self._apply_all:
            index = list(range(len(self.data.shapes)))
        else:
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
        if self._apply_all:
            index = list(range(len(self.data.shapes)))
        else:
            index = self.selected_shapes
        for i in index:
            self.data.update_opacity(i, opacity)
        self.refresh()
        self.events.opacity()

    @property
    def apply_all(self):
        """bool: whether to apply a manipulation to all shapes or just the
        selected ones
        """
        return self._apply_all

    @apply_all.setter
    def apply_all(self, apply_all):
        self._apply_all = apply_all
        self.events.apply_all()

    @property
    def selected_shapes(self):
        """list: list of currently selected shapes
        """
        return self._selected_shapes

    @selected_shapes.setter
    def selected_shapes(self, selected_shapes):
        self._selected_shapes = selected_shapes
        self._selected_box = self.select_box(selected_shapes)

    @property
    def mode(self):
        """None, str: Interactive mode
        """
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode == self.mode:
            return
        old_mode = self.mode
        if mode == 'select':
            self.cursor = 'pointing'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom, press <delete> to remove selected'
            self.status = mode
            self._mode = mode
        elif mode == 'direct':
            self.cursor = 'pointing'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom, press <delete> to remove selected'
            self.status = mode
            self._mode = mode
        elif mode == 'pan/zoom':
            self.cursor = 'standard'
            self.interactive = True
            self.help = ''
            self.status = mode
            self._mode = mode
        elif mode == 'add_rectangle':
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom'
            self.status = mode
            self._mode = mode
        elif mode == 'add_ellipse':
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom'
            self.status = mode
            self._mode = mode
        elif mode == 'add_line':
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom'
            self.status = mode
            self._mode = mode
        elif mode == 'add_path':
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom, press <esc> to finish drawing'
            self.status = mode
            self._mode = mode
        elif mode == 'add_polygon':
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom, press <esc> to finish drawing'
            self.status = mode
            self._mode = mode
        elif mode == 'vertex_insert':
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom'
            self.status = mode
            self._mode = mode
        elif mode == 'vertex_remove':
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom'
            self.status = mode
            self._mode = mode
        else:
            raise ValueError("Mode not recongnized")

        self.events.mode(mode=mode)
        if (mode == 'vertex_insert' or mode == 'vertex_remove' or
              mode == 'direct' or mode == 'select') and (old_mode == 'vertex_insert' or
              old_mode == 'vertex_remove' or old_mode == 'direct' or old_mode == 'select'):
            pass
        else:
            self._finish_drawing()
        self.refresh()

    def _get_shape(self):
        if len(self.data._vertices) == 0:
            return [1, 1]
        else:
            return np.max(self.data._vertices, axis=0) + 1

    def _update(self):
        """Update the underlying visual.
        """
        if self._need_display_update:
            self._need_display_update = False
            self._set_view_slice(self.viewer.dims.indices)

        if self._need_visual_update:
            self._need_visual_update = False
            self._node.update()

    def _refresh(self):
        """Fully refresh the underlying visual.
        """
        self._need_display_update = True
        self._update()

    def _set_view_slice(self, indices):
        """Sets the view given the indices to slice with.
        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.
        """
        faces = self.data._mesh_triangles[self.data._mesh_triangles_z_order]
        colors = self.data._mesh_triangles_colors[self.data._mesh_triangles_z_order]
        vertices = self.data._mesh_vertices
        if len(faces) == 0:
            self._node._subvisuals[3].set_data(vertices=None, faces=None)
        else:
            self._node._subvisuals[3].set_data(vertices=vertices, faces=faces,
                                               face_colors=colors)
        self._need_visual_update = True
        self._set_highlight()
        self._update()

    def select_box(self, index=True):
        if index is True:
            box = create_box(self.data._vertices)
        elif isinstance(index, (list, np.ndarray)):
            if len(index) == 0:
                box = None
            elif len(index) == 1:
                box = copy(self.data.shapes[index[0]]._box)
            else:
                box = create_box(self.data._vertices[np.isin(self.data._index, index)])
        else:
            box = copy(self.data.shapes[index]._box)
        if box is not None:
            rot = box[1]
            length_box = np.linalg.norm(box[6] - box[0])
            if length_box > 0:
                rot = rot-self._rotion_handle_length*(box[6] - box[0])/length_box
            box = np.append(box, [rot], axis=0)
        return box

    def _set_highlight(self):
        if self._hover_shape is not None or len(self.selected_shapes) > 0:
            # show outlines hover shape or any selected shapes
            if len(self.selected_shapes) > 0:
                index = copy(self.selected_shapes)
                if self._hover_shape is not None:
                    if self._hover_shape in index:
                        pass
                    else:
                        index.append(self._hover_shape)
                index.sort()
                meshes = self.data._mesh_triangles_index
                faces_indices = [i for i, x in enumerate(meshes) if x[0] in index and x[1] == 1]
                meshes = self.data._mesh_vertices_index
                vertices_indices = [i for i, x in enumerate(meshes) if x[0] in index and x[1] == 1]
            else:
                index = self._hover_shape
                faces_indices = np.all(self.data._mesh_triangles_index == [index, 1], axis=1)
                faces_indices = np.where(faces_indices)[0]
                vertices_indices = np.all(self.data._mesh_vertices_index == [index, 1], axis=1)
                vertices_indices = np.where(vertices_indices)[0]

            vertices = (self.data._mesh_vertices_centers[vertices_indices] +
                        self._highlight_width*self.data._mesh_vertices_offsets[vertices_indices])
            faces = self.data._mesh_triangles[faces_indices]

            if type(index) is list:
                faces_index = self.data._mesh_triangles_index[faces_indices][:, 0]
                starts = np.unique(self.data._mesh_vertices_index[vertices_indices][:, 0], return_index=True)[1]
                for i, ind in enumerate(index):
                    faces[faces_index == ind] = faces[faces_index == ind] - vertices_indices[starts[i]] + starts[i]
            else:
                faces = faces - vertices_indices[0]
            self._node._subvisuals[2].set_data(vertices=vertices, faces=faces,
                                               color=self._highlight_color)
        else:
            self._node._subvisuals[2].set_data(vertices=None, faces=None)

        if len(self.selected_shapes) > 0:
            if self.mode == 'select':
                inds = list(range(0, 8))
                inds.append(9)
                box = self._selected_box[inds]
                if self._hover_shape is None:
                    face_color = 'white'
                elif self._hover_vertex is None:
                    face_color = 'white'
                else:
                    face_color = self._highlight_color
                edge_color = self._highlight_color
                self._node._subvisuals[0].set_data(box, size=self._vertex_size,
                                                   face_color=face_color,
                                                   edge_color=edge_color,
                                                   edge_width=1.5,
                                                   symbol='square',
                                                   scaling=False)
                self._node._subvisuals[1].set_data(pos=box[[1, 2, 4, 6, 0, 1, 8]],
                                                   color=edge_color, width=1.5)
            elif (self.mode == 'direct' or self.mode == 'add_path' or
                  self.mode == 'add_polygon' or self.mode == 'add_rectangle' or
                  self.mode == 'add_ellipse' or self.mode == 'add_line' or
                  self.mode == 'vertex_insert' or self.mode == 'vertex_remove'):
                inds = np.isin(self.data._index, self.selected_shapes)
                vertices = self.data._vertices[inds]
                # If currently adding path don't show box over last vertex
                if self.mode == 'add_path':
                    vertices = vertices[:-1]

                if self._hover_shape is None:
                    face_color = 'white'
                elif self._hover_vertex is None:
                    face_color = 'white'
                else:
                    face_color = self._highlight_color
                edge_color = self._highlight_color
                self._node._subvisuals[0].set_data(vertices,
                                                   size=self._vertex_size,
                                                   face_color=face_color,
                                                   edge_color=edge_color,
                                                   edge_width=1.5,
                                                   symbol='square',
                                                   scaling=False)
                self._node._subvisuals[1].set_data(pos=None, width=0)
        elif self._is_selecting:
            box = create_box(self._drag_box)
            edge_color = self._highlight_color
            self._node._subvisuals[0].set_data(np.empty((0, 2)), size=0)
            self._node._subvisuals[1].set_data(pos=box[[0, 2, 4, 6, 0]],
                                               color=edge_color, width=1.5)
        else:
            self._node._subvisuals[0].set_data(np.empty((0, 2)), size=0)
            self._node._subvisuals[1].set_data(pos=None, width=0)

    def _select(self):
        if (self.selected_shapes == self._selected_shapes_stored and
                self._hover_shape == self._hover_shape_stored and
                self._hover_vertex == self._hover_vertex_stored):
            return
        self._selected_shapes_stored = copy(self.selected_shapes)
        self._hover_shape_stored = copy(self._hover_shape)
        self._hover_vertex_stored = copy(self._hover_vertex)
        self._set_highlight()

    def _finish_drawing(self):
        """Resets variables involed in shape drawing so that a new shape can
        then be drawn
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
        if self._creating is True and self.mode == 'add_path':
            vertices = self.data._vertices[self.data._index == index]
            if len(vertices) <= 2:
                self.data.remove(index)
            else:
                self.data.edit(index, vertices[:-1])
        if self._creating is True and self.mode == 'add_polygon':
            vertices = self.data._vertices[self.data._index == index]
            if len(vertices) <= 2:
                self.data.remove(index)
        self._creating = False
        self.refresh()

    def remove_selected(self):
        """Removes any selected shapes
        """
        to_remove = np.sort(self.selected_shapes)[::-1]
        for index in to_remove:
            self.data.remove(index)
        self.selected_shapes = []
        shape, vertex = self._shape_at(self._mouse_coord)
        self._hover_shape = shape
        self._hover_vertex = vertex
        self.status = self.get_message(self._mouse_coord, shape, vertex)
        self.refresh()

    def _rotate_box(self, angle, center=[0, 0]):
        """Perfroms a rotation on the selected box

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
        box = np.matmul(box, transform.T)
        self._selected_box = box + center

    def _scale_box(self, scale, center=[0, 0]):
        """Perfroms a scaling on the selected box

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
        if not np.all(box[1] == box[9]):
            r = self._rotion_handle_length
            box[9] = box[1] + r*(box[9]-box[1])/np.linalg.norm(box[9]-box[1])
        self._selected_box = box + center

    def _transform_box(self, transform, center=[0, 0]):
        """Perfroms a linear transformation on the selected box

        Parameters
        ----------
        transform : np.ndarray
            2x2 array specifying linear transform.
        center : list
            coordinates of center of rotation.
        """
        box = self._selected_box - center
        box = np.matmul(box, transform.T)
        if not np.all(box[1] == box[9]):
            r = self._rotion_handle_length
            box[9] = box[1] + r*(box[9]-box[1])/np.linalg.norm(box[9]-box[1])
        self._selected_box = box + center

    def _shape_at(self, coord):
        """Determines if any shapes at given coord by looking inside triangle
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
            if self.mode == 'select':
                # Check if inside vertex of bounding box or rotation handle
                inds = list(range(0, 8))
                inds.append(9)
                box = self._selected_box[inds]
                distances = abs(box - coord[:2])

                # Get the vertex sizes
                scene = self.viewer._canvas.scene
                transform = scene.node_transform(self._node)
                rescale = transform.map([1, 1])[:2] - transform.map([0, 0])[:2]
                sizes = self._vertex_size*rescale.mean()/2

                # Check if any matching vertices
                matches = np.all(distances <= sizes, axis=1).nonzero()
                if len(matches[0]) > 0:
                    return self.selected_shapes[0], matches[0][-1]
            elif (self.mode == 'direct' or self.mode == 'vertex_insert' or
                  self.mode == 'vertex_remove'):
                # Check if inside vertex of shape
                inds = np.isin(self.data._index, self.selected_shapes)
                vertices = self.data._vertices[inds]
                distances = abs(vertices - coord[:2])

                # Get the vertex sizes
                scene = self.viewer._canvas.scene
                transform = scene.node_transform(self._node)
                rescale = transform.map([1, 1])[:2] - transform.map([0, 0])[:2]
                sizes = self._vertex_size*rescale.mean()/2

                # Check if any matching vertices
                matches = np.all(distances <= sizes, axis=1).nonzero()[0]
                if len(matches) > 0:
                    index = inds.nonzero()[0][matches[-1]]
                    shape = self.data._index[index]
                    _, idx = np.unique(self.data._index, return_index=True)
                    return shape, index - idx[shape]

        # Check if mouse inside shape
        triangles = self.data._mesh_vertices[self.data._mesh_triangles]
        inside = inside_triangles(triangles - coord[:2])
        shapes = self.data._mesh_triangles_index[inside]

        if len(shapes) > 0:
            indices = shapes[:, 0]
            z_list = self.data._z_order.tolist()
            order_indices = np.array([z_list.index(m) for m in indices])
            ordered_shapes = indices[np.argsort(order_indices)]
            return ordered_shapes[0], None
        else:
            return None, None

    def _shapes_in_box(self, box):
        box = create_box(box)[[0, 4]]
        triangles = self.data._mesh_vertices[self.data._mesh_triangles]

        # check if triangle corners are inside box
        points_inside = np.empty(triangles.shape[:-1], dtype=bool)
        for i in range(3):
            inside = ([box[1] >= triangles[:, 0, :], triangles[:, i, :] >= box[0]])
            points_inside[:, i] = np.all(np.concatenate(inside, axis=1), axis=1)

        # check if triangle edges intersect box edges
        # NOT IMPLEMENTED

        inside = np.any(points_inside, axis=1)
        shapes = self.data._mesh_triangles_index[inside, 0]

        return np.unique(shapes).tolist()

    def _get_coord(self, position):
        """Converts the position of the cursor in canvas coordinates to image
        coordinates

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
        pos = [pos[1], pos[0]]
        coord = np.array([pos[1], pos[0]])
        self._mouse_coord = coord
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

    def _move(self, coord):
        """Moves object at given mouse position and set of indices.

        Parameters
        ----------
        coord : sequence of two int
            Position of mouse cursor in image coordinates.
        """
        vertex = self._moving_vertex
        if (self.mode == 'select' or self.mode == 'add_rectangle' or
                self.mode == 'add_ellipse' or self.mode == 'add_line'):
            if len(self.selected_shapes) > 0:
                self._is_moving = True
                if vertex is None:
                    # Check where dragging box from to move whole object
                    if self._drag_start is None:
                        center = self._selected_box[-1]
                        self._drag_start = coord - center
                    center = self._selected_box[-1]
                    shift = coord - center - self._drag_start
                    for index in self.selected_shapes:
                        self.data.shift(index, shift)
                    self._selected_box = self._selected_box + shift
                    self.refresh()
                elif vertex < 8:
                    # Corner / edge vertex is being dragged so resize object
                    box = self._selected_box
                    if self._fixed_vertex is None:
                        self._fixed_index = np.mod(vertex+4, 8)
                        self._fixed_vertex = box[self._fixed_index]

                    size = (box[np.mod(self._fixed_index+4, 8)] -
                            box[self._fixed_index])
                    offset = box[-1] - box[-2]
                    offset = offset/np.linalg.norm(offset)
                    offset_perp = np.array([offset[1], -offset[0]])

                    if np.mod(self._fixed_index, 2) == 0:
                        # corner selected
                        fixed = self._fixed_vertex
                        new = copy(coord)
                        if self._fixed_aspect:
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
                        dist = np.dot(new-fixed, offset)/np.dot(size, offset)
                        dist_perp = (np.dot(new-fixed, offset_perp) /
                                     np.dot(size, offset_perp))
                        scale = np.array([dist_perp, dist])
                    elif np.mod(self._fixed_index, 4) == 1:
                        # top selected
                        fixed = self._fixed_vertex
                        new = copy(coord)
                        dist = np.dot(new-fixed, offset)/np.dot(size, offset)
                        scale = np.array([1, dist])
                    else:
                        # side selected
                        fixed = self._fixed_vertex
                        new = copy(coord)
                        dist_perp = (np.dot(new-fixed, offset_perp) /
                                     np.dot(size, offset_perp))
                        scale = np.array([dist_perp, 1])

                    # prevent box from shrinking below a threshold size
                    scene = self.viewer._canvas.scene
                    transform = scene.node_transform(self._node)
                    rescale = (transform.map([1, 1])[:2] -
                               transform.map([0, 0])[:2])
                    threshold = self._vertex_size*rescale.mean()/8
                    scale[abs(scale*size) < threshold] = 1

                    # check orientation of box
                    angle = -np.arctan2(offset[0], -offset[1])
                    if angle == 0:
                        for index in self.selected_shapes:
                            self.data.scale(index, scale,
                                            center=self._fixed_vertex)
                        self._scale_box(scale, center=self._fixed_vertex)
                    else:
                        rotation = np.array([[np.cos(angle), np.sin(angle)],
                                            [-np.sin(angle), np.cos(angle)]])
                        scale_mat = np.array([[scale[0], 0], [0, scale[1]]])
                        inv_rot = np.array([[np.cos(angle), -np.sin(angle)],
                                           [np.sin(angle), np.cos(angle)]])
                        transform = np.matmul(rotation,
                                              np.matmul(scale_mat, inv_rot))
                        for index in self.selected_shapes:
                            self.data.shift(index, -self._fixed_vertex)
                            self.data.transform(index, transform)
                            self.data.shift(index, self._fixed_vertex)
                        self._transform_box(transform,
                                            center=self._fixed_vertex)
                    self.refresh()
                elif vertex == 8:
                    # Rotation handle is being dragged so rotate object
                    handle = self._selected_box[-1]
                    if self._drag_start is None:
                        self._fixed_vertex = self._selected_box[-2]
                        offset = handle - self._fixed_vertex
                        self._drag_start = -np.arctan2(offset[0],
                                                       -offset[1])/np.pi*180

                    new_offset = coord - self._fixed_vertex
                    new_angle = -np.arctan2(new_offset[0],
                                            -new_offset[1])/np.pi*180
                    fixed_offset = handle - self._fixed_vertex
                    fixed_angle = -np.arctan2(fixed_offset[0],
                                              -fixed_offset[1])/np.pi*180

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
        elif (self.mode == 'direct' or self.mode == 'add_path' or
              self.mode == 'add_polygon'):
            if len(self.selected_shapes) > 0:
                if vertex is not None:
                    self._is_moving = True
                    index = self._moving_shape
                    object_type = self.data.shapes[index].shape_type
                    if object_type == 'ellipse':
                        # Direct vertex moving of ellipse not implemented
                        pass
                    else:
                        indices = self.data._index == index
                        vertices = self.data._vertices[indices]
                        vertices[vertex] = coord
                        self.data.edit(index, vertices)
                        shapes = self.selected_shapes
                        self._selected_box = self.select_box(shapes)
                        self.refresh()
            else:
                self._is_selecting = True
                if self._drag_start is None:
                    self._drag_start = coord
                self._drag_box = np.array([self._drag_start, coord])
                self._set_highlight()
        elif self.mode == 'vertex_insert' or self.mode == 'vertex_remove':
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

        if self.mode == 'pan/zoom':
            # If in pan/zoom mode do nothing
            pass
        elif self.mode == 'select':
            if not self._is_moving and not self._is_selecting:
                shape, vertex = self._shape_at(coord)
                self._moving_shape = shape
                self._moving_vertex = vertex
                if vertex is None:
                    if shift and shape is not None:
                        if shape in self.selected_shapes:
                            self.selected_shapes.remove(shape)
                            shapes = self.selected_shapes
                            self._selected_box = self.select_box(shapes)
                        else:
                            self.selected_shapes.append(shape)
                            shapes = self.selected_shapes
                            self._selected_box = self.select_box(shapes)
                    elif shape is not None:
                        if shape not in self.selected_shapes:
                            self.selected_shapes = [shape]
                    else:
                        self.selected_shapes = []
                self._select()
                self.status = self.get_message(coord, shape, vertex)
        elif self.mode == 'direct':
            if not self._is_moving and not self._is_selecting:
                shape, vertex = self._shape_at(coord)
                self._moving_shape = shape
                self._moving_vertex = vertex
                if vertex is None:
                    if shift and shape is not None:
                        if shape in self.selected_shapes:
                            self.selected_shapes.remove(shape)
                            shapes = self.selected_shapes
                            self._selected_box = self.select_box(shapes)
                        else:
                            self.selected_shapes.append(shape)
                            shapes = self.selected_shapes
                            self._selected_box = self.select_box(shapes)
                    elif shape is not None:
                        if shape not in self.selected_shapes:
                            self.selected_shapes = [shape]
                    else:
                        self.selected_shapes = []
                self._select()
                self.status = self.get_message(coord, shape, vertex)
        elif (self.mode == 'add_rectangle' or self.mode == 'add_ellipse' or
              self.mode == 'add_line'):
            # Start drawing a rectangle / ellipse / line
            transform = self.viewer._canvas.scene.node_transform(self._node)
            rescale = transform.map([1, 1])[:2] - transform.map([0, 0])[:2]
            size = self._vertex_size*rescale.mean()/4
            new_z_index = max(self.data._z_index, default=-1) + 1
            if self.mode == 'add_rectangle':
                data = np.array([coord, coord+size])
                self.data.add(data, shape_type='rectangle',
                              edge_width=self.edge_width,
                              edge_color=self.edge_color,
                              face_color=self.face_color,
                              opacity=self.opacity,
                              z_index=new_z_index)
            elif self.mode == 'add_ellipse':
                data = np.array([coord+size/2, [size, size]])
                self.data.add(data, shape_type='ellipse',
                              edge_width=self.edge_width,
                              edge_color=self.edge_color,
                              face_color=self.face_color,
                              opacity=self.opacity,
                              z_index=new_z_index)
            elif self.mode == 'add_line':
                data = np.array([coord, coord+size])
                self.data.add(data, shape_type='line',
                              edge_width=self.edge_width,
                              edge_color=self.edge_color,
                              face_color=self.face_color,
                              opacity=self.opacity,
                              z_index=new_z_index)
            else:
                raise ValueError("Mode not recongnized")
            self.selected_shapes = [len(self.data.shapes)-1]
            ind = 4
            self._moving_shape = self.selected_shapes[0]
            self._moving_vertex = ind
            self._hover_shape = self.selected_shapes[0]
            self._hover_vertex = ind
            self._creating = True
            self._select()
            self.refresh()
        elif (self.mode == 'add_path' or self.mode == 'add_polygon'):
            if self._creating is False:
                # Start drawing a path
                data = np.array([coord, coord])
                new_z_index = max(self.data._z_index, default=-1) + 1
                self.data.add(data, shape_type='path',
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
                self._creating = True
                self._select()
            else:
                # Add to an existing path or polygon
                index = self._moving_shape
                if self.mode == 'add_polygon':
                    self.data.shapes[index].shape_type = 'polygon'
                vertices = self.data._vertices[self.data._index == index]
                vertices = np.concatenate((vertices, [coord]),  axis=0)
                # Change the selected vertex
                self._moving_vertex = self._moving_vertex + 1
                self._hover_vertex = self._hover_vertex + 1
                self.data.edit(index, vertices)
                self._selected_box = self.select_box(self.selected_shapes)
            self.status = self.get_message(coord, self._hover_shape,
                                           self._hover_vertex)
        elif self.mode == 'vertex_insert':
            if len(self.selected_shapes) == 0:
                # If none selected return immediately
                return

            all_lines = np.empty((0, 2, 2))
            all_lines_shape = np.empty((0, 2), dtype=int)
            for index in self.selected_shapes:
                object_type = self.data.shapes[index].shape_type
                if object_type == 'ellipse':
                    # Adding vertex to ellipse not implemented
                    pass
                else:
                    vertices = self.data._vertices[self.data._index == index]
                    # Find which edge new vertex should inserted along
                    closed = object_type != 'path'
                    n = len(vertices)
                    if closed:
                        lines = np.array([[vertices[i],
                                         vertices[np.mod(i+1, n)]] for i in
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
            object_type = self.data.shapes[index].shape_type
            if object_type == 'line':
                # Adding vertex to path turns it into line
                object_type = 'path'
                self.data.shapes[index].shape_type = object_type
            closed = object_type != 'path'
            vertices = self.data._vertices[self.data._index == index]
            if closed is not True:
                if int(ind) == 1 and loc < 0:
                    ind = 0
                elif int(ind) == len(vertices)-1 and loc > 1:
                    ind = ind + 1

            vertices = np.insert(vertices, ind, [coord], axis=0)
            with self.freeze_refresh():
                self.data.edit(index, vertices)
                self._selected_box = self.select_box(self.selected_shapes)
            shape, vertex = self._shape_at(coord)
            self._hover_shape = shape
            self._hover_vertex = vertex
            self.refresh()
            self.status = self.get_message(coord, shape, vertex)
        elif self.mode == 'vertex_remove':
            shape, vertex = self._shape_at(coord)
            if vertex is not None:
                # have clicked on a current vertex so remove
                index = shape
                object_type = self.data.shapes[index].shape_type
                if object_type == 'ellipse':
                    # Removing vertex from ellipse not implemented
                    return
                vertices = self.data._vertices[self.data._index == index]
                if len(vertices) <= 2:
                    # If only 2 vertices present, remove whole shape
                    with self.freeze_refresh():
                        self.remove_shapes(index=index)
                else:
                    if object_type == 'polygon' and len(vertices) == 3:
                        self.data.shapes[index].shape_type = 'path'
                    # Remove clicked on vertex
                    vertices = np.delete(vertices, vertex, axis=0)
                    with self.freeze_refresh():
                        self.data.edit(index, vertices)
                        shapes = self.selected_shapes
                        self._selected_box = self.select_box(shapes)
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

        if self.mode == 'pan/zoom':
            # If in pan/zoom mode just look at coord all
            shape, vertex = self._shape_at(coord)
        elif self.mode == 'select':
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
                self._select()
            shape = self._hover_shape
            vertex = self._hover_vertex
        elif self.mode == 'direct':
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
                self._select()
            shape = self._hover_shape
            vertex = self._hover_vertex
        elif (self.mode == 'add_rectangle' or self.mode == 'add_ellipse' or
              self.mode == 'add_line'):
            # While drawing a shape or doing nothing
            if self._creating and event.is_dragging:
                # Drag any selected shapes
                self._move(coord)
                shape = self._hover_shape
                vertex = self._hover_vertex
            else:
                shape, vertex = self._shape_at(coord)
        elif (self.mode == 'add_path' or self.mode == 'add_polygon'):
            # While drawing a path or doing nothing
            if self._creating:
                # Drag any selected shapes
                self._move(coord)
                shape = self._hover_shape
                vertex = self._hover_vertex
            else:
                shape, vertex = self._shape_at(coord)
        elif (self.mode == 'vertex_insert' or self.mode == 'vertex_remove'):
            self._hover_shape, self._hover_vertex = self._shape_at(coord)
            self._select()
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

        if self.mode == 'pan/zoom':
            # If in pan/zoom mode do nothing
            pass
        elif self.mode == 'select':
            shape, vertex = self._shape_at(coord)
            if not self._is_moving and not self._is_selecting and not shift:
                if shape is not None:
                    self.selected_shapes = [shape]
                else:
                    self.selected_shapes = []
            elif self._is_selecting:
                self.selected_shapes = self._shapes_in_box(self._drag_box)
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
            self._select()
            self.status = self.get_message(coord, shape, vertex)
        elif self.mode == 'direct':
            shape, vertex = self._shape_at(coord)
            if not self._is_moving and not self._is_selecting and not shift:
                if shape is not None:
                    self.selected_shapes = [shape]
                else:
                    self.selected_shapes = []
            elif self._is_selecting:
                self.selected_shapes = self._shapes_in_box(self._drag_box)
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
            self._select()
            self.status = self.get_message(coord, shape, vertex)
        elif (self.mode == 'add_rectangle' or self.mode == 'add_ellipse' or
                self.mode == 'add_line'):
            self._finish_drawing()
            shape, vertex = self._shape_at(coord)
            self.status = self.get_message(coord, shape, vertex)
        elif (self.mode == 'add_path' or self.mode == 'add_polygon'):
            pass
        elif (self.mode == 'vertex_insert' or self.mode == 'vertex_remove'):
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
                if self.mode != 'pan/zoom':
                    self._mode_history = self.mode
                    self._selected_shapes_history = copy(self.selected_shapes)
                    self.mode = 'pan/zoom'
                else:
                    self._mode_history = 'pan/zoom'
            elif event.key == 'Shift':
                self._fixed_aspect = True
                box = self._selected_box
                if box is not None:
                    size = box[4]-box[0]
                    if not np.any(size == np.zeros(2)):
                        self._aspect_ratio = abs(size[1] / size[0])
                    else:
                        self._aspect_ratio = 1
                else:
                    self._aspect_ratio = 1
                if self._is_moving:
                    self._move(self._mouse_coord)
            elif event.key == 'r':
                self.mode = 'add_rectangle'
            elif event.key == 'e':
                self.mode = 'add_ellipse'
            elif event.key == 'l':
                self.mode = 'add_line'
            elif event.key == 't':
                self.mode = 'add_path'
            elif event.key == 'p':
                self.mode = 'add_polygon'
            elif event.key == 'd':
                self.mode = 'direct'
            elif event.key == 's':
                self.mode = 'select'
            elif event.key == 'z':
                self.mode = 'pan/zoom'
            elif event.key == 'v':
                self.mode = 'vertex_insert'
            elif event.key == 'x':
                self.mode = 'vertex_remove'
            elif event.key == 'a':
                if (self.mode == 'direct' or self.mode == 'select' or
                        self.mode == 'vertex_insert' or
                        self.mode == 'vertex_remove'):
                    self.selected_shapes = list(range(len(self.data.shapes)))
                    self._select()
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
            if self._mode_history != 'pan/zoom':
                self.mode = self._mode_history
                self.selected_shapes = self._selected_shapes_history
                self._select()
        elif event.key == 'Shift':
            self._fixed_aspect = False
            if self._is_moving:
                self._move(self._mouse_coord)
