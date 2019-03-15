from typing import Union
from collections import Iterable

import numpy as np
from numpy import clip, integer, ndarray, append, insert, delete, empty
from copy import copy

from ...util import is_permutation
from .shape_utils import create_box, inside_boxes, inside_triangles, point_to_lines
from ...util.event import Event
from .._base_layer import Layer
from .._register import add_to_viewer
from ..._vispy.scene.visuals import Mesh
from ..._vispy.scene.visuals import Markers
from ..._vispy.scene.visuals import Line
from ..._vispy.scene.visuals import Compound as VisualNode
from .utils import ShapesData
from vispy.color import get_color_names, Color

from .view import QtShapesLayer
from .view import QtShapesControls
from .shapes_list import ShapesList

@add_to_viewer
class Shapes(Layer):
    """Shapes layer.
    Parameters
    ----------
    shapes : list
        List of dictionaries, where each dictionary corresponds to one shape
        and has the keys corresponding to the keyword args of the Shapes class.
        `shape_type` and `data` are required. `edge_width`, `edge_color`,
        `face_color`, and `z_order` are optional.
    name : str, keyword-only
        Name of the layer.
    """

    def __init__(self, shapes, *, name=None):

        visual = VisualNode([Markers(), Line(), Mesh(), Mesh()])

        super().__init__(visual, name)

        # Freeze refreshes
        with self.freeze_refresh():
            # Save the style params
            self.data = ShapesList(shapes)
            self._colors = get_color_names()
            self.apply_all = True
            self.edge_width = 1
            self.edge_color = 'black'
            self.face_color = 'white'
            #self.opacity = 1

            self._vertex_size = 10

            # update flags
            self._need_display_update = False
            self._need_visual_update = False

            self._highlight = True
            self._highlight_color = (0, 0.6, 1)
            self._highlight_thickness = 0.5
            self._rotion_handle_length = 20
            self._selected_shapes = []
            self._selected_shapes_stored = []
            self._selected_shapes_history = []
            self._selected_box = None
            self._hover_shapes = [None, None]
            self._hover_shapes_stored = [None, None]

            self._drag_start = None
            self._fixed_vertex = None
            self._fixed_aspect = False
            self._selected_vertex = [None, None]
            self._aspect_ratio = 1
            self._is_moving=False
            self._fixed_index = 0
            self._is_selecting = False
            self._drag_box = None
            self._mouse_coord = [0, 0]

            self._ready_to_create = False
            self._creating = False
            self._create_coord = [None, None]
            self._prefixed_size = np.array([10, 10])

            self._mode = 'pan/zoom'
            self._mode_history = self._mode
            self._status = self._mode

            self.events.add(mode=Event)
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
        # if self._apply_all:
        #     index = True
        # else:
        #     index = self.selected_shapes
        # self.set_thickness(index=index, thickness=self._edge_width)


    @property
    def edge_color(self):
        """Color, ColorArray: color of edges and lines
        """

        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color):
        self._edge_color = edge_color
        # if self._apply_all:
        #     index = True
        # else:
        #     index = self.selected_shapes
        # self.set_color(index=index, edge_color=self._edge_color)

    @property
    def face_color(self):
        """Color, ColorArray: color of faces
        """

        return self._face_color

    @face_color.setter
    def face_color(self, face_color):
        self._face_color = face_color
        # if self._apply_all:
        #     index = True
        # else:
        #     index = self.selected_shapes
        # self.set_color(index=index, face_color=self._face_color)

    # @property
    # def opacity(self):
    #     """float: Opacity value between 0.0 and 1.0.
    #     """
    #     return self._opacity
    #
    # @opacity.setter
    # def opacity(self, opacity):
    #     if not 0.0 <= opacity <= 1.0:
    #         raise ValueError('opacity must be between 0.0 and 1.0; '
    #                          f'got {opacity}')
    #
    #     self._opacity = opacity
    #     if self._apply_all:
    #         index = True
    #     else:
    #         index = self.selected_shapes
    #     self.set_opacity(index=index, opacity=opacity)
    #     self.refresh()
    #     self.events.opacity()

    @property
    def apply_all(self):
        """bool: whether to apply gui manipulations to all shapes or just selected
        """

        return self._apply_all

    @apply_all.setter
    def apply_all(self, apply_all):
        self._apply_all = apply_all

    @property
    def selected_shapes(self):
        """list: list of currently selected shapes
        """

        return self._selected_shapes

    @selected_shapes.setter
    def selected_shapes(self, selected_shapes):
        self._selected_shapes = selected_shapes
        self._selected_box = self.select_box(selected_shapes)

    # @property
    # def z_order(self):
    #     """list: list of z order of objects. If there are N objects it
    #     must be a permutation of 0,...,N-1
    #     """
    #
    #     return self._z_order
    #
    # @z_order.setter
    # def z_order(self, z_order):
    #     ## Check z_order is a permutation of 0,...,N-1
    #     if not is_permutation(z_order, self.data.count):
    #         raise ValueError('z_order is not permutation')
    #
    #     self._z_order = np.array(z_order)
    #
    #     if len(self._z_order) == 0:
    #         self._z_order_faces = np.empty((0), dtype=int)
    #     else:
    #         _, idx = np.unique(self.data._mesh_triangles_index[:,0], return_index=True)
    #         z_order_faces = [np.arange(idx[z], idx[z]+self._face_counts[z]) for z in self._z_order]
    #         self._z_order_faces = np.concatenate(z_order_faces)
    #     self.refresh()

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
            self.help = 'hold <space> to pan/zoom'
            self.status = mode
            self._mode = mode
        elif mode == 'add_polygon':
            self.cursor = 'cross'
            self.interactive = False
            self.help = 'hold <space> to pan/zoom'
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
        faces = self.data._mesh_triangles
        colors = self.data._mesh_triangles_colors
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
        if self._highlight and (self._hover_shapes[0] is not None or len(self.selected_shapes)>0):
            # show outlines hover shape or any selected shapes
            if len(self.selected_shapes)>0:
                index = copy(self.selected_shapes)
                if self._hover_shapes[0] is not None:
                    if self._hover_shapes[0] in index:
                        pass
                    else:
                        index.append(self._hover_shapes[0])
                index.sort()
            else:
                index = self._hover_shapes[0]

            faces_indices = self.data._select_meshes(index, self.data._mesh_triangles_index, 1)
            vertices_indices = self.data._select_meshes(index, self.data._mesh_vertices_index, 1)
            vertices = self.data._mesh_vertices[vertices_indices]
            faces = self.data._mesh_triangles[faces_indices]
            if type(index) is list:
                faces_index = self.data._mesh_triangles_index[faces_indices][:,0]
                starts = np.unique(self.data._mesh_vertices_index[vertices_indices][:,0], return_index=True)[1]
                for i, ind in enumerate(index):
                    faces[faces_index==ind] = faces[faces_index==ind] - vertices_indices[starts[i]] + starts[i]
            else:
                faces = faces - vertices_indices[0]
            self._node._subvisuals[2].set_data(vertices=vertices, faces=faces,
                                               color=self._highlight_color)
        else:
            self._node._subvisuals[2].set_data(vertices=None, faces=None)

        if self._highlight and len(self.selected_shapes) > 0:
            if self.mode == 'select':
                inds = list(range(0,8))
                inds.append(9)
                box = self._selected_box[inds]
                if self._hover_shapes[0] is None:
                    face_color = 'white'
                elif self._hover_shapes[1] is None:
                    face_color = 'white'
                else:
                    face_color = self._highlight_color
                edge_color = self._highlight_color
                self._node._subvisuals[0].set_data(box, size=self._vertex_size, face_color=face_color,
                                                   edge_color=edge_color, edge_width=1.5,
                                                   symbol='square', scaling=False)
                self._node._subvisuals[1].set_data(pos=box[[1, 2, 4, 6, 0, 1, 8]],
                                                   color=edge_color, width=1.5)
            elif (self.mode == 'direct' or self.mode == 'add_path' or
                  self.mode == 'add_polygon' or self.mode == 'add_rectangle' or
                  self.mode == 'add_ellipse' or self.mode == 'add_line' or
                  self.mode == 'vertex_insert' or self.mode == 'vertex_remove'):
                inds = np.isin(self.data.index, self.selected_shapes)
                vertices = self.data.vertices[inds]
                # If currently adding path don't show box over last vertex
                if self.mode == 'add_path':
                    vertices = vertices[:-1]

                if self._hover_shapes[0] is None:
                    face_color = 'white'
                elif self._hover_shapes[1] is None:
                    face_color = 'white'
                else:
                    face_color = self._highlight_color
                edge_color = self._highlight_color
                self._node._subvisuals[0].set_data(vertices, size=self._vertex_size, face_color=face_color,
                                                   edge_color=edge_color, edge_width=1.5,
                                                   symbol='square', scaling=False)
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
            self._hover_shapes == self._hover_shapes_stored):
            return
        self._highlight = True
        self._selected_shapes_stored = copy(self.selected_shapes)
        self._hover_shapes_stored = copy(self._hover_shapes)
        self._set_highlight()

    def _unselect(self):
        if self._highlight:
            self._highlight = False
            self._selected_shapes_stored = []
            self._hover_shapes_stored = [None, None]
            self._set_highlight()

    def _finish_drawing(self):
        index = self._selected_vertex[0]
        self._ready_to_create = False
        self._create_coord = [None, None]
        self._is_moving = False
        self.selected_shapes = []
        self._drag_start = None
        self._drag_box = None
        self._fixed_vertex = None
        self._selected_vertex = [None, None]
        self._hover_shapes = [None, None]
        if self._creating is True and self.mode == 'add_path':
            vertices = self.data._vertices[self.data._index==index]
            if len(vertices) <= 2:
                self.data.remove_one_shape(index)
            else:
                self.data.edit(index, vertices[:-1])
        if self._creating is True and self.mode == 'add_polygon':
            vertices = self.data._vertices[self.data._index==index]
            if len(vertices) <= 2:
                self.data.remove_one_shape(index)
        self._creating = False
        self._unselect()

    def _shape_at(self, indices):
        """Determines if any shapes at given indices by looking inside triangle
        meshes.
        Parameters
        ----------
        indices : sequence of int
            Indices to check if shape at.
        """
        # Check selected shapes
        if len(self.selected_shapes) > 0:
            if self.mode == 'select':
                # Check if inside vertex of bounding box or rotation handle
                inds = list(range(0,8))
                inds.append(9)
                box = self._selected_box[inds]
                distances = abs(box - indices[:2])

                # Get the vertex sizes
                transform = self.viewer._canvas.scene.node_transform(self._node)
                rescale = (transform.map([1, 1])[:2] - transform.map([0, 0])[:2]).mean()
                sizes = self._vertex_size*rescale/2

                # Check if any matching vertices
                matches = np.all(distances <=  sizes, axis=1).nonzero()
                if len(matches[0]) > 0:
                    return [self.selected_shapes[0], matches[0][-1]]
            elif (self.mode == 'direct' or self.mode == 'vertex_insert' or
                  self.mode == 'vertex_remove'):
                # Check if inside vertex of shape
                inds = np.isin(self.data.index, self.selected_shapes)
                vertices = self.data.vertices[inds]
                distances = abs(vertices - indices[:2])

                # Get the vertex sizes
                transform = self.viewer._canvas.scene.node_transform(self._node)
                rescale = (transform.map([1, 1])[:2] - transform.map([0, 0])[:2]).mean()
                sizes = self._vertex_size*rescale/2

                # Check if any matching vertices
                matches = np.all(distances <=  sizes, axis=1).nonzero()[0]
                if len(matches) > 0:
                    index = inds.nonzero()[0][matches[-1]]
                    shape = self.data.index[index]
                    _, idx = np.unique(self.data.index, return_index=True)
                    return [shape, index - idx[shape]]

        # Check if mouse inside shape
        triangles = self.data._mesh_vertices[self.data._mesh_triangles]
        shapes = self.data._mesh_triangles_index[inside_triangles(triangles - indices[:2])]

        if len(shapes) > 0:
            indices = shapes[:, 0]
            # z_list = self._z_order.tolist()
            # order_indices = np.array([z_list.index(m) for m in indices])
            # ordered_shapes = indices[np.argsort(order_indices)]
            ordered_shapes = indices
            return [ordered_shapes[0], None]
        else:
            return [None, None]

    def _shapes_in_box(self, box):
        box = create_box(box)[[0, 4]]
        triangles = self.data._mesh_vertices[self.data._mesh_triangles]

        # check if triangle corners are inside box
        points_inside = np.empty(triangles.shape[:-1], dtype=bool)
        for i in range(3):
            points_inside[:, i] = np.all(np.concatenate(([box[1] >= triangles[:,0,:], triangles[:,i,:] >= box[0]]), axis=1), axis=1)

        # check if triangle edges intersect box edges
        # NOT IMPLEMENTED

        inside = np.any(points_inside, axis=1)
        shapes = self.data._mesh_triangles_index[inside, 0]

        return np.unique(shapes).tolist()

    def _get_coord(self, position, indices):
        max_shape = self.viewer.dims.max_shape
        transform = self.viewer._canvas.scene.node_transform(self._node)
        pos = transform.map(position)
        pos = [pos[1], pos[0]]
        coord = copy(indices)
        coord[0] = pos[1]
        coord[1] = pos[0]
        self._mouse_coord = np.array(coord)
        return self._mouse_coord

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
        coord_shift = copy(coord)
        coord_shift[0] = int(coord[1])
        coord_shift[1] = int(coord[0])
        msg = f'{coord_shift.astype(int)}, {self.name}'
        if value[0] is not None:
            msg = msg + ', shape ' + str(value[0])
            if value[1] is not None:
                msg = msg + ', vertex ' + str(value[1])
        return msg







    def remove_selected(self):
        """ Removes any currently selected shapes if in `select` or `direct`
        mode
        """
        if self.mode == 'select' or self.mode == 'direct':
            to_delete = copy(self.selected_shapes)
            self.selected_shapes = []
            self._hover_shapes = [None, None]
            #self.remove_shapes(to_delete)

    #
    # def scale_selected(self, scale, vertex=-2):
    #     """Perfroms a scaling on selected shapes
    #     Parameters
    #     ----------
    #     scale : float, list
    #         scalar or list specifying rescaling of shapes in 2D.
    #     vertex : int
    #         coordinate of bounding box to use as center of scaling.
    #     index : bool, list, int
    #         index of objects to be selected. Where True corresponds to all
    #         objects, a list of integers to a list of objects, and a single
    #         integer to that particular object.
    #     """
    #     self.data.select_box(index)
    #     center = self._selected_box[vertex]
    #     self.data.scale_shapes(scale, center=center, index=index)
    #     self.refresh()
    #
    # def flip_vertical_shapes(self, vertex=-2, index=True):
    #     """Perfroms an vertical flip on selected shapes
    #     Parameters
    #     ----------
    #     vertex : int
    #         coordinate of bounding box to use as center of flip axes.
    #     index : bool, list, int
    #         index of objects to be selected. Where True corresponds to all
    #         objects, a list of integers to a list of objects, and a single
    #         integer to that particular object.
    #     """
    #     self.data.select_box(index)
    #     center = self._selected_box[vertex]
    #     self.data.flip_vertical_shapes(center=center, index=index)
    #     self.refresh()
    #
    # def flip_horizontal_shapes(self, vertex=-2, index=True):
    #     """Perfroms an horizontal flip on selected shapes
    #     Parameters
    #     ----------
    #     vertex : int
    #         coordinate of bounding box to use as center of flip axes.
    #     index : bool, list, int
    #         index of objects to be selected. Where True corresponds to all
    #         objects, a list of integers to a list of objects, and a single
    #         integer to that particular object.
    #     """
    #     self.data.select_box(index)
    #     center = self._selected_box[vertex]
    #     self.data.flip_horizontal_shapes(center=center, index=index)
    #     self.refresh()
    #
    # def rotate_shapes(self, angle, vertex=-2, index=True):
    #     """Perfroms a rotation on selected shapes
    #     Parameters
    #     ----------
    #     angle : float
    #         angle specifying rotation of shapes in degrees.
    #     vertex : int
    #         coordinate of bounding box to use as center of rotation.
    #     index : bool, list, int
    #         index of objects to be selected. Where True corresponds to all
    #         objects, a list of integers to a list of objects, and a single
    #         integer to that particular object.
    #     """
    #     self.data.select_box(index)
    #     center = self._selected_box[vertex]
    #     self.data.rotate_shapes(angle, center=center, index=index)
    #     self.refresh()
    #
    # def shift_shapes(self, shift, index=True):
    #     """Perfroms an 2D shift on selected shapes
    #     Parameters
    #     ----------
    #     shift : np.ndarray
    #         length 2 array specifying shift of shapes.
    #     index : bool, list, int
    #         index of objects to be selected. Where True corresponds to all
    #         objects, a list of integers to a list of objects, and a single
    #         integer to that particular object.
    #     """
    #     self.data.shift_shapes(shift, index=index)
    #     self.refresh()
    #
    # def set_thickness(self, index=True, thickness=1):
    #     if isinstance(thickness, (list, np.ndarray)):
    #         if index is True:
    #             self.data._thickness = thickness
    #         else:
    #             self.data._thickness[index] = thickness
    #     else:
    #         if index is True:
    #             self.data._thickness = np.repeat(thickness, len(self.data._thickness))
    #         elif isinstance(index, (list, np.ndarray)):
    #             self.data._thickness[index] = np.repeat(thickness, len(index))
    #         else:
    #             self.data._thickness[index] = thickness
    #     self.data.update_thickness(index)
    #     self.refresh()
    #
    # def move_forward(self, index):
    #     if type(index) is list:
    #         z_order = self._z_order.tolist()
    #         indices = [z_order.index(i) for i in index]
    #         for i in np.sort(indices):
    #             self._move_one_forward(z_order[i])
    #     else:
    #         self._move_one_forward(index)
    #     self.z_order = self._z_order
    #
    # def move_backward(self, index):
    #     if type(index) is list:
    #         z_order = self._z_order.tolist()
    #         indices = [z_order.index(i) for i in index]
    #         for i in np.sort(indices)[::-1]:
    #             self._move_one_backward(z_order[i])
    #     else:
    #         self._move_one_backward(index)
    #     self.z_order = self._z_order
    #
    # def move_to_front(self, index):
    #     if type(index) is list:
    #         z_order = self._z_order.tolist()
    #         indices = [z_order.index(i) for i in index]
    #         for i in np.sort(indices)[::-1]:
    #             self._move_one_to_front(z_order[i])
    #     else:
    #         self._move_one_to_front(index)
    #     self.z_order = self._z_order
    #
    # def move_to_back(self, index):
    #     if type(index) is list:
    #         z_order = self._z_order.tolist()
    #         indices = [z_order.index(i) for i in index]
    #         for i in np.sort(indices):
    #             self._move_one_to_back(z_order[i])
    #     else:
    #         self._move_one_to_back(index)
    #     self.z_order = self._z_order
    #
    # def _move_one_forward(self, index):
    #     ind = self._z_order.tolist().index(index)
    #     if ind != 0:
    #         self._z_order[ind] = self._z_order[ind-1]
    #         self._z_order[ind-1] = index
    #
    # def _move_one_backward(self, index):
    #     ind = self._z_order.tolist().index(index)
    #     if ind != len(self._z_order)-1:
    #         self._z_order[ind] = self._z_order[ind+1]
    #         self._z_order[ind+1] = index
    #
    # def _move_one_to_front(self, index):
    #     self._z_order[1:] = self._z_order[self._z_order!=index]
    #     self._z_order[0] = index
    #
    # def _move_one_to_back(self, index):
    #     self._z_order[:-1] = self._z_order[self._z_order!=index]
    #     self._z_order[-1] = index
    #
    #
    # def set_opacity(self, index=True, opacity=1):
    #     if type(opacity) is list:
    #         if index is True:
    #             for i, op in enumerate(opacity):
    #                 self._face_color_array[i, 3] = op
    #                 self._edge_color_array[i, 3] = op
    #             for i in range(len(self.data._mesh_triangles_index)):
    #                 self._mesh_color_array[i] = self._face_color_array[self.data._mesh_triangles_index[i, 0]]
    #         else:
    #             assert(type(index) is list and len(opacity)==len(index))
    #             for i, ind in enumerate(index):
    #                 indices = self.data._select_meshes(ind, self.data._mesh_triangles_index)
    #                 self._face_color_array[ind, 3] = op[i]
    #                 self._edge_color_array[ind, 3] = op[i]
    #                 self._mesh_color_array[indices] = op[i]
    #     else:
    #         indices = self.data._select_meshes(index, self.data._mesh_triangles_index)
    #         self._mesh_color_array[indices, 3] = opacity
    #         if index is True:
    #             for i in range(self.data.count):
    #                 self._face_color_array[i, 3] = opacity
    #                 self._edge_color_array[i, 3] = opacity
    #         else:
    #             self._face_color_array[index, 3] = opacity
    #             self._edge_color_array[index, 3] = opacity
    #
    # def set_color(self, index=True, edge_color=False, face_color=False):
    #     if face_color is False:
    #         pass
    #     else:
    #         if type(face_color) is list:
    #             if index is True:
    #                 for i, col in enumerate(face_color):
    #                     self._face_color_array[i] = Color(col).rgba
    #                 for i in range(len(self.data._mesh_triangles_index)):
    #                     if self.data._mesh_triangles_index[i, 2] == 0:
    #                         self._mesh_color_array[i] = self._face_color_array[self.data._mesh_triangles_index[i, 0]]
    #             else:
    #                 assert(type(index) is list and len(face_color)==len(index))
    #                 for i, ind in enumerate(index):
    #                     indices = self.data._select_meshes(ind, self.data._mesh_triangles_index, 0)
    #                     self._face_color_array[ind] = Color(face_color[i]).rgba
    #                     self._mesh_color_array[indices] = self._face_color_array[ind]
    #         else:
    #             indices = self.data._select_meshes(index, self.data._mesh_triangles_index, 0)
    #             self._mesh_color_array[indices] = Color(face_color).rgba
    #             if index is True:
    #                 for i in range(self.data.count):
    #                     self._face_color_array[i] = Color(face_color).rgba
    #             else:
    #                 self._face_color_array[index] = Color(face_color).rgba
    #     if edge_color is False:
    #         pass
    #     else:
    #         if type(edge_color) is list:
    #             if index is True:
    #                 for i, col in enumerate(edge_color):
    #                     self._edge_color_array[i] = Color(col).rgba
    #                 for i in range(len(self.data._mesh_triangles_index)):
    #                     if self.data._mesh_triangles_index[i, 2] == 1:
    #                         self._mesh_color_array[i] = self._edge_color_array[self.data._mesh_triangles_index[i, 0]]
    #             else:
    #                 assert(type(index) is list and len(face_color)==len(index))
    #                 for i, ind in enumerate(index):
    #                     indices = self.data._select_meshes(ind, self.data._mesh_triangles_index, 1)
    #                     self._edge_color_array[ind] = Color(edge_color[i]).rgba
    #                     self._mesh_color_array[indices] = self._edge_color_array[ind]
    #         else:
    #             indices = self.data._select_meshes(index, self.data._mesh_triangles_index, 1)
    #             self._mesh_color_array[indices] = Color(edge_color).rgba
    #             if index is True:
    #                 for i in range(self.data.count):
    #                     self._edge_color_array[i] = Color(edge_color).rgba
    #             else:
    #                 self._edge_color_array[index] = Color(edge_color).rgba
    #
    #     self.refresh()
    #
    def _move(self, coord):
        """Moves object at given mouse position
        and set of indices.
        Parameters
        ----------
        coord : sequence of two int
            Position of mouse cursor in data.
        """
        index = self.selected_shapes
        vertex = self._selected_vertex[1]
        if (self.mode == 'select' or self.mode == 'add_rectangle' or
        self.mode == 'add_ellipse' or self.mode == 'add_line'):
            if len(index) > 0:
                self._is_moving=True
                if vertex is None:
                    #Check where dragging box from to move whole object
                    if self._drag_start is None:
                        center = self._selected_box[-1]
                        self._drag_start = coord - center
                    center = self._selected_box[-1]
                    shift = coord - center - self._drag_start
                    self.shift_shapes(shift, index=index)
                elif vertex < 8:
                    #Corner / edge vertex is being dragged so resize object
                    box = self._selected_box
                    if self._fixed_vertex is None:
                        self._fixed_index = np.mod(vertex+4,8)
                        self._fixed_vertex = box[self._fixed_index]
                        if np.any(box[4]-box[0] == np.zeros(2)):
                            self._aspect_ratio = 1
                        else:
                            self._aspect_ratio = (box[4][1]-box[0][1])/(box[4][0]-box[0][0])

                    size = box[np.mod(self._fixed_index+4,8)] - box[self._fixed_index]
                    offset = box[-1] - box[-2]
                    offset = offset/np.linalg.norm(offset)
                    offset_perp = np.array([offset[1], -offset[0]])

                    if np.mod(self._fixed_index, 2) == 0:
                        # corner selected
                        fixed = self._fixed_vertex
                        new = copy(coord)
                        if self._fixed_aspect:
                            ratio = abs((new - fixed)[1]/(new - fixed)[0])
                            if ratio>self._aspect_ratio:
                                new[1] = fixed[1]+(new[1]-fixed[1])*self._aspect_ratio/ratio
                            else:
                                new[0] = fixed[0]+(new[0]-fixed[0])*ratio/self._aspect_ratio
                        dist = np.dot(new-fixed, offset)/np.dot(size, offset)
                        dist_perp = np.dot(new-fixed, offset_perp)/np.dot(size, offset_perp)
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
                        dist_perp = np.dot(new-fixed, offset_perp)/np.dot(size, offset_perp)
                        scale = np.array([dist_perp, 1])

                    # prvent box from dissappearing if shrunk near 0
                    scale[scale==0]=1

                    # check orientation of box
                    angle = -np.arctan2(offset[0], -offset[1])
                    if angle == 0:
                        self.data.scale_shapes(scale, center=self._fixed_vertex, index=index)
                    else:
                        rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
                        scale_tranform = np.array([[scale[0], 0], [0, scale[1]]])
                        inverse_rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                        transform = np.matmul(rotation, np.matmul(scale_tranform, inverse_rotation))
                        self.data.shift_shapes(-self._fixed_vertex, index=index)
                        self.data.transform_shapes(transform, index=index)
                        self.data.shift_shapes(self._fixed_vertex, index=index)
                    self.refresh()
                elif vertex==8:
                    #Rotation handle is being dragged so rotate object
                    handle = self._selected_box[-1]
                    if self._drag_start is None:
                        self._fixed_vertex = self._selected_box[-2]
                        offset = handle - self._fixed_vertex
                        self._drag_start = -np.arctan2(offset[0], -offset[1])/np.pi*180

                    new_offset = coord - self._fixed_vertex
                    new_angle = -np.arctan2(new_offset[0], -new_offset[1])/np.pi*180
                    fixed_offset = handle - self._fixed_vertex
                    fixed_angle = -np.arctan2(fixed_offset[0], -fixed_offset[1])/np.pi*180

                    if np.linalg.norm(new_offset)<1:
                        angle = 0
                    elif self._fixed_aspect:
                        angle = np.round(new_angle/45)*45 - fixed_angle
                    else:
                        angle = new_angle - fixed_angle

                    self.data.rotate_shapes(angle, center=self._fixed_vertex, index=index)
                    self.refresh()
            else:
                self._is_selecting=True
                if self._drag_start is None:
                    self._drag_start = coord
                self._drag_box = np.array([self._drag_start, coord])
                self._set_highlight()
        elif (self.mode == 'direct' or self.mode == 'add_path' or
              self.mode == 'add_polygon'):
            if len(index) > 0:
                if vertex is not None:
                    self._is_moving=True
                    index = self._selected_vertex[0]
                    vertices = self.data.vertices[self.data.index == index]
                    vertices[vertex] = coord
                    self.edit_shape(index, vertices)
            else:
                self._is_selecting=True
                if self._drag_start is None:
                    self._drag_start = coord
                self._drag_box = np.array([self._drag_start, coord])
                self._set_highlight()
        elif self.mode == 'vertex_insert' or self.mode == 'vertex_remove':
            if len(index) > 0:
                pass
            else:
                self._is_selecting=True
                if self._drag_start is None:
                    self._drag_start = coord
                self._drag_box = np.array([self._drag_start, coord])
                self._set_highlight()


    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.
        """
        position = event.pos
        indices = self.viewer.dims.indices
        coord = self._get_coord(position, indices)
        shift = 'Shift' in event.modifiers

        if self.mode == 'pan/zoom':
            # If in pan/zoom mode do nothing
            pass
        elif self.mode == 'select':
            if not self._is_moving and not self._is_selecting:
                shape = self._shape_at(coord)
                self._selected_vertex = shape
                if self._selected_vertex[1] is None:
                    if shift and shape[0] is not None:
                        if shape[0] in self.selected_shapes:
                            self.selected_shapes.remove(shape[0])
                            self._selected_box = self.select_box(self.selected_shapes)
                        else:
                            self.selected_shapes.append(shape[0])
                            self._selected_box = self.select_box(self.selected_shapes)
                    elif shape[0] is not None:
                        if shape[0] not in self.selected_shapes:
                            self.selected_shapes = [shape[0]]
                    else:
                        self.selected_shapes = []
                self._select()
                self.status = self.get_message(coord, shape)
        elif self.mode == 'direct':
            if not self._is_moving and not self._is_selecting:
                shape = self._shape_at(coord)
                self._selected_vertex = shape
                if self._selected_vertex[1] is None:
                    if shift and shape[0] is not None:
                        if shape[0] in self.selected_shapes:
                            self.selected_shapes.remove(shape[0])
                            self._selected_box = self.select_box(self.selected_shapes)
                        else:
                            self.selected_shapes.append(shape[0])
                            self._selected_box = self.select_box(self.selected_shapes)
                    elif shape[0] is not None:
                        if shape[0] not in self.selected_shapes:
                            self.selected_shapes = [shape[0]]
                    else:
                        self.selected_shapes = []
                self._select()
                self.status = self.get_message(coord, shape)
        elif (self.mode == 'add_rectangle' or self.mode == 'add_ellipse' or
              self.mode == 'add_line'):
            # Start drawing a rectangle / ellipse / line
            self._ready_to_create = True
            self._create_coord = coord
        elif (self.mode == 'add_path' or self.mode == 'add_polygon'):
            if self._creating is False:
                # Start drawing a path
                shape = np.array([[coord, coord]])
                self.add_shapes(paths = shape, thickness=self.edge_width)
                self.selected_shapes = [self.data.count-1]
                ind = 1
                self._selected_vertex = [self.selected_shapes[0], ind]
                self._hover_shapes = [self.selected_shapes[0], ind]
                self._creating = True
                self._select()
            else:
                # Add to an existing path or polygon
                index = self._selected_vertex[0]
                if self.mode == 'add_polygon':
                    self.data.id[index] = self.data.objects.index('polygon')
                vertices = self.data.vertices[self.data.index==index]
                vertices = np.concatenate((vertices, [coord]),  axis=0)
                # Change the selected vertex
                self._selected_vertex[1] = self._selected_vertex[1]+1
                self._hover_shapes[1] = self._hover_shapes[1]+1
                self.edit_shape(self._selected_vertex[0], vertices)
            self.status = self.get_message(coord, self._hover_shapes)
        elif self.mode == 'vertex_insert':
            if len(self.selected_shapes) == 0:
                #If none selected return immediately
                return

            all_lines = np.empty((0, 2, 2))
            all_lines_shape = np.empty((0, 2), dtype=int)
            for shape in self.selected_shapes:
                object_type = self.data.id[shape]
                if object_type == self.data.objects.index('ellipse'):
                    # Adding vertex to ellipse not implemented
                    pass
                else:
                    vertices = self.data.vertices[self.data.index==shape]
                    # Find which edge new vertex should inserted along
                    closed = object_type != self.data.objects.index('path')
                    n = len(vertices)
                    if closed:
                        lines = np.array([[vertices[i], vertices[np.mod(i+1, n)]] for i in range(n)])
                    else:
                        lines = np.array([[vertices[i], vertices[i+1]] for i in range(n-1)])
                    all_lines = np.concatenate((all_lines, lines), axis=0)
                    indices = np.array([np.repeat(shape, len(lines)), list(range(len(lines)))]).T
                    all_lines_shape = np.concatenate((all_lines_shape, indices), axis=0)
            if len(all_lines) == 0:
                # No appropriate shapes found
                return

            ind, loc = point_to_lines(coord, all_lines)
            index = all_lines_shape[ind][0]
            ind = all_lines_shape[ind][1]+1
            object_type = self.data.id[shape]
            if object_type == self.data.objects.index('line'):
                # Adding vertex to path turns it into line
                object_type = self.data.objects.index('path')
                self.data.id[index] = object_type
            closed = object_type != self.data.objects.index('path')
            vertices = self.data.vertices[self.data.index==index]
            if closed is not True:
                if int(ind) == 1 and loc < 0:
                    ind = 0
                elif int(ind) == len(vertices)-1 and loc > 1:
                    ind = ind + 1

            vertices = np.insert(vertices, ind, [coord], axis=0)
            with self.freeze_refresh():
                self.edit_shape(index, vertices)
            shape = self._shape_at(coord)
            self._hover_shapes = shape
            self.refresh()
            self.status = self.get_message(coord, shape)
        elif self.mode == 'vertex_remove':
            shape = self._shape_at(coord)
            if shape[1] is not None:
                # have clicked on a current vertex so remove
                index = shape[0]
                vertex = shape[1]
                object_type = self.data.id[index]
                if object_type == self.data.objects.index('ellipse'):
                    # Removing vertex from ellipse not implemented
                    return
                vertices = self.data.vertices[self.data.index==index]
                if len(vertices) <= 2:
                    # If only 2 vertices present, remove whole shape
                    with self.freeze_refresh():
                        self.remove_shapes(index=index)
                else:
                    if (object_type == self.data.objects.index('polygon') and
                        len(vertices) == 3):
                        self.data.id[index] = self.data.objects.index('path')
                    # Remove clicked on vertex
                    vertices = np.delete(vertices, vertex, axis=0)
                    with self.freeze_refresh():
                        self.edit_shape(index, vertices)
                shape = self._shape_at(coord)
                self._hover_shapes = shape
                self.refresh()
                self.status = self.get_message(coord, shape)
        else:
            raise ValueError("Mode not recongnized")

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        if event.pos is None:
            return
        position = event.pos
        indices = self.viewer.dims.indices
        coord = self._get_coord(position, indices)

        if self.mode == 'pan/zoom':
            # If in pan/zoom mode just look at coord all
            shape = self._shape_at(coord)
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
                self._hover_shapes = self._shape_at(coord)
                self._select()
            shape = self._hover_shapes
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
                self._hover_shapes = self._shape_at(coord)
                self._select()
            shape = self._hover_shapes
        elif (self.mode == 'add_rectangle' or self.mode == 'add_ellipse' or
              self.mode == 'add_line'):
            # If ready to create rectangle, ellipse or line start making one
            if self._ready_to_create and np.all(self._create_coord != coord):
                shape = np.array([[self._create_coord, coord]])
                if self.mode == 'add_rectangle':
                    self.add_shapes(rectangles = shape, thickness=self.edge_width)
                elif self.mode == 'add_ellipse':
                    self.add_shapes(ellipses = shape, thickness=self.edge_width)
                elif self.mode == 'add_line':
                    self.add_shapes(lines = shape, thickness=self.edge_width)
                else:
                    raise ValueError("Mode not recongnized")
                self._ready_to_create = False
                self.selected_shapes = [self.data.count-1]
                ind = np.all(self._selected_box==coord, axis=1).nonzero()[0][0]
                self._selected_vertex = [self.selected_shapes[0], ind]
                self._hover_shapes = [self.selected_shapes[0], ind]
                self._creating = True
                self._select()
            # While drawing a shape or doing nothing
            if self._creating and event.is_dragging:
                # Drag any selected shapes
                self._move(coord)
                shape = self._hover_shapes
            else:
                shape = self._shape_at(coord)
        elif (self.mode == 'add_path' or self.mode == 'add_polygon'):
            # While drawing a path or doing nothing
            if self._creating:
                # Drag any selected shapes
                self._move(coord)
                shape = self._hover_shapes
            else:
                shape = self._shape_at(coord)
        elif (self.mode == 'vertex_insert' or self.mode == 'vertex_remove'):
            self._hover_shapes = self._shape_at(coord)
            self._select()
            shape = self._hover_shapes
        else:
            raise ValueError("Mode not recongnized")

        self.status = self.get_message(coord, shape)

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.
        """
        position = event.pos
        indices = self.viewer.dims.indices
        coord = self._get_coord(position, indices)
        shift = 'Shift' in event.modifiers

        if self.mode == 'pan/zoom':
            # If in pan/zoom mode do nothing
            pass
        elif self.mode == 'select':
            shape = self._shape_at(coord)
            if not self._is_moving and not self._is_selecting and not shift:
                if shape[0] is not None:
                    self.selected_shapes = [shape[0]]
                else:
                    self.selected_shapes = []
            elif self._is_selecting:
                self.selected_shapes = self._shapes_in_box(self._drag_box)
                self._is_selecting=False
                self._set_highlight()
            self._is_moving = False
            self._drag_start = None
            self._drag_box = None
            self._fixed_vertex = None
            self._selected_vertex = [None, None]
            self._hover_shapes = shape
            self._select()
            self.status = self.get_message(coord, shape)
        elif self.mode == 'direct':
            shape = self._shape_at(coord)
            if not self._is_moving and not self._is_selecting and not shift:
                if shape[0] is not None:
                    self.selected_shapes = [shape[0]]
                else:
                    self.selected_shapes = []
            elif self._is_selecting:
                self.selected_shapes = self._shapes_in_box(self._drag_box)
                self._is_selecting=False
                self._set_highlight()
            self._is_moving = False
            self._drag_start = None
            self._drag_box = None
            self._fixed_vertex = None
            self._selected_vertex = [None, None]
            self._hover_shapes = shape
            self._select()
            self.status = self.get_message(coord, shape)
        elif (self.mode == 'add_rectangle' or self.mode == 'add_ellipse' or
             self.mode == 'add_line'):
            # Finish drawing a rectangle, ellipse or line
            if self._ready_to_create:
                shape = np.array([[self._create_coord-self._prefixed_size/2,
                                   self._create_coord+self._prefixed_size/2]])
                if self.mode == 'add_rectangle':
                    self.add_shapes(rectangles = shape, thickness=self.edge_width)
                elif self.mode == 'add_ellipse':
                    self.add_shapes(ellipses = shape, thickness=self.edge_width)
                elif self.mode == 'add_line':
                    self.add_shapes(lines = shape, thickness=self.edge_width)
                else:
                    raise ValueError("Mode not recongnized")
                self._ready_to_create = False
                shape = [self.data.count-1, None]
            else:
                self._finish_drawing()
                shape = self._shape_at(coord)
            self.status = self.get_message(coord, shape)
        elif (self.mode == 'add_path' or self.mode == 'add_polygon'):
            pass
        elif (self.mode == 'vertex_insert' or self.mode == 'vertex_remove'):
            pass
        else:
            raise ValueError("Mode not recongnized")

    def on_key_press(self, event):
        """Called whenever key pressed in canvas.
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
                if self._is_moving:
                    box = self._selected_box
                    if box is not None:
                        self._aspect_ratio = abs((box[4][1]-box[0][1])/(box[4][0]-box[0][0]))
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
                    self.mode == 'vertex_insert' or self.mode == 'vertex_remove'):
                    self.selected_shapes = list(range(self.data.count))
                    self._select()
            elif event.key == 'Backspace':
                self.remove_selected()
            elif event.key == 'Escape':
                self._finish_drawing()

    def on_key_release(self, event):
        """Called whenever key released in canvas.
        """
        if event.key == ' ':
            if self._mode_history != 'pan/zoom':
                self.mode = self._mode_history
                self.selected_shapes = self._selected_shapes_history
                self._select()
        elif event.key == 'Shift':
            self._fixed_aspect = False
            if self._is_moving:
                box = self._selected_box
                if box is not None:
                    self._aspect_ratio = abs((box[4][1]-box[0][1])/(box[4][0]-box[0][0]))
                self._move(self._mouse_coord)
