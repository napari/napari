from typing import Union
from collections import Iterable

import numpy as np
from numpy import clip, integer, ndarray, append, insert, delete, empty
from copy import copy

from ..util import is_permutation, inside_boxes, inside_triangles
from ._base_layer import Layer
from ._register import add_to_viewer
from .._vispy.scene.visuals import Mesh
from .._vispy.scene.visuals import Markers
from .._vispy.scene.visuals import Line
from .._vispy.scene.visuals import Compound as VisualNode
from .shapes_data import ShapesData
from vispy.color import get_color_names, Color

from .qt import QtShapesLayer

@add_to_viewer
class Shapes(Layer):
    """Shapes layer.
    Parameters
    ----------
    lines : np.ndarray
        Nx2x2 array of endpoints of lines.
    rectangles : np.ndarray
        Nx2x2 array of corners of rectangles.
    ellipses : np.ndarray
        Nx2x2 array of corners of ellipses.
    paths : list
        list of Nx2 arrays of points on each path.
    polygons : list
        list of Nx2 arrays of vertices of each polygon.
    edge_width : int
        width of all lines and edges in pixels.
    face_color : Color, ColorArray
        fill color of all faces
    edge_color : Color, ColorArray
        color of all lines and edges
    """

    def __init__(self, points=None, lines=None, paths=None, rectangles=None,
                 ellipses=None, polygons=None, edge_width=1, edge_color='black',
                 face_color='white'):

        visual = VisualNode([Markers(), Line(), Mesh(), Mesh()])

        super().__init__(visual)
        self.name = 'shapes'

        # Save the style params
        self._edge_width = edge_width
        self._edge_color = edge_color
        self._face_color = face_color
        self._colors = get_color_names()

        # Save the shape data
        self._data = ShapesData(lines=lines, paths=paths, rectangles=rectangles,
                                ellipses=ellipses, polygons=polygons,
                                thickness=self._edge_width)

        c = Color(self.edge_color).rgba
        self._color_array = np.array([c for i in range(len(self.data._mesh_faces))])
        faces_indices = self.data._mesh_faces_index[:,2]==0
        self._color_array[faces_indices] = Color(self.face_color).rgba

        self._show_faces = np.ones(len(self.data._mesh_faces), dtype=bool)
        default_z_order, counts = np.unique(self.data._mesh_faces_index[:,0], return_counts=True)
        self._object_counts = counts
        self._z_order = default_z_order
        self._z_order_faces = np.arange(len(self.data._mesh_faces_index))

        self._vertex_size = 10

        # update flags
        self._need_display_update = False
        self._need_visual_update = False

        self._highlight = True
        self._highlight_color = (0, 0.6, 1)
        self._highlight_thickness = 0.5
        self._selected_shapes = []
        self._selected_shapes_stored = []
        self._hover_shapes = [None, None]
        self._hover_shapes_stored = [None, None]

        self._drag_start = None
        self._fixed_vertex = None
        self._fixed_aspect = False
        self._selected_vertex = None
        self._aspect_ratio = 1
        self._is_moving=False
        self._fixed_index = 0

        # self._ready_to_create_box = False
        # self._creating_box = False
        # self._create_tl = None

        self._qt = QtShapesLayer(self)

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
        self.set_thickness(thickness=self._edge_width)

    @property
    def edge_color(self):
        """Color, ColorArray: color of edges and lines
        """

        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color):
        self._edge_color = edge_color
        self.set_color(edge_color=self._edge_color)

    @property
    def face_color(self):
        """Color, ColorArray: color of faces
        """

        return self._face_color

    @face_color.setter
    def face_color(self, face_color):
        self._face_color = face_color
        self.set_color(face_color=self._face_color)

    @property
    def z_order(self):
        """list: list of z order of objects. If there are N objects it
        must be a permutation of 0,...,N-1
        """

        return self._z_order

    @z_order.setter
    def z_order(self, z_order):
        ## Check z_order is a permutation of 0,...,N-1
        assert(is_permutation(z_order, len(self._object_counts)))

        self._z_order = np.array(z_order)

        if len(self._z_order) == 0:
            self._z_order_faces = np.empty((0), dtype=int)
        else:
            offsets = np.zeros(len(self._object_counts) + 1, dtype=int)
            offsets[1:] = self._object_counts.cumsum()
            z_order_faces = [np.arange(offsets[z], offsets[z]+self._object_counts[z]) for z in self._z_order]
            self._z_order_faces = np.concatenate(z_order_faces)

        self.refresh()

    def _get_shape(self):
        return [1, 1]

    # def _get_shape(self):
    #     if len(self.coords) == 0:
    #         return np.ones(self.coords.shape[2:],dtype=int)
    #     else:
    #         return np.max(self.coords, axis=(0,1)) + 1

    def _update(self):
        """Update the underlying visual.
        """
        if self._need_display_update:
            self._need_display_update = False
            self._set_view_slice(self.viewer.dimensions.indices)

        if self._need_visual_update:
            self._need_visual_update = False
            self._node.update()

    def _refresh(self):
        """Fully refresh the underlying visual.
        """
        self._need_display_update = True
        self._update()

    def add_shapes(self, lines=None, rectangles=None, ellipses=None, paths=None,
                   polygons=None, thickness=1):
        """Adds shapes to the exisiting ones.
        """
        num_shapes = len(self.data.id)
        num_faces = len(self.data._mesh_faces_index)
        self.data.add_shapes(lines=lines, rectangles=rectangles,
                             ellipses=ellipses, paths=paths, polygons=polygons,
                             thickness=thickness)
        new_num_shapes = len(self.data.id)
        t = np.ones(len(self.data._mesh_faces_index)-len(self._show_faces), dtype=bool)
        self._show_faces = np.append(self._show_faces, t)

        ec = Color(self.edge_color).rgba
        fc = Color(self.face_color).rgba
        color_array = np.repeat([ec], len(self.data._mesh_faces_index)-num_faces, axis=0)
        color_array[self.data._mesh_faces_index[num_faces:,2]==0] = fc
        self._color_array = np.concatenate((self._color_array, color_array), axis=0)
        default_z_order, counts = np.unique(self.data._mesh_faces_index[:,0], return_counts=True)
        self._object_counts = counts
        self.z_order = np.concatenate((np.arange(num_shapes, new_num_shapes), self.z_order))

    def set_shapes(self, lines=None, rectangles=None, ellipses=None, paths=None,
                   polygons=None, thickness=1):
        """Resets shapes to be only these ones.
        """
        self.data.set_shapes(lines=lines, rectangles=rectangles,
                             ellipses=ellipses, paths=paths, polygons=polygons,
                             thickness=thickness)

        self._show_faces = np.ones(len(self.data._mesh_faces_index), dtype=bool)

        ec = Color(self.edge_color).rgba
        fc = Color(self.face_color).rgba
        color_array = np.repeat([ec], len(self.data._mesh_faces_index), axis=0)
        color_array[self.data._mesh_faces_index[:,2]==0] = fc
        self._color_array = np.array(color_array)
        default_z_order, counts = np.unique(self.data._mesh_faces_index[:,0], return_counts=True)
        self._object_counts = counts
        self.z_order = default_z_order

    def remove_shapes(self, index=True):
        """Remove shapes specified in index.
        """
        if index==True:
            self.data.remove_all_shapes()
            self._show_faces = np.empty((0), dtype=bool)
            self._color_array =  np.empty((0, 4))
            self._object_counts = np.empty((0), dtype=int)
            self.z_order = np.empty((0), dtype=int)
        elif type(index) is list:
            for i in np.sort(index)[::-1]:
                self._remove_one_shape(int(i))
            self.z_order = self._z_order
        else:
            self._remove_one_shape(index)
            self.z_order = self._z_order

    def _remove_one_shape(self, index):
        assert(type(index) is int)
        faces_indices = self.data._mesh_faces_index[:, 0]
        self.data.remove_one_shape(index)
        z_order = self._z_order[self._z_order!=index]
        z_order[z_order>index] = z_order[z_order>index]-1
        self._z_order = z_order
        self._object_counts = np.delete(self._object_counts, index, axis=0)
        self._show_faces = self._show_faces[faces_indices!=index]
        self._color_array = self._color_array[faces_indices!=index]

    def scale_shapes(self, scale, vertex=-1, index=True):
        """Perfroms a scaling on selected shapes
        Parameters
        ----------
        scale : float, list
            scalar or list specifying rescaling of shapes in 2D.
        vertex : int
            coordinate of bounding box to use as center of scaling.
        index : bool, list, int
            index of objects to be selected. Where True corresponds to all
            objects, a list of integers to a list of objects, and a single
            integer to that particular object.
        """
        center = self.data.selected_box(index)[vertex]
        self.data.scale_shapes(scale, center=center, index=index)
        self.refresh()

    def flip_vertical_shapes(self, vertex=-1, index=True):
        """Perfroms an vertical flip on selected shapes
        Parameters
        ----------
        vertex : int
            coordinate of bounding box to use as center of flip axes.
        index : bool, list, int
            index of objects to be selected. Where True corresponds to all
            objects, a list of integers to a list of objects, and a single
            integer to that particular object.
        """
        center = self.data.selected_box(index)[vertex]
        self.data.flip_vertical_shapes(center=center, index=index)
        self.refresh()

    def flip_horizontal_shapes(self, vertex=-1, index=True):
        """Perfroms an horizontal flip on selected shapes
        Parameters
        ----------
        vertex : int
            coordinate of bounding box to use as center of flip axes.
        index : bool, list, int
            index of objects to be selected. Where True corresponds to all
            objects, a list of integers to a list of objects, and a single
            integer to that particular object.
        """
        center = self.data.selected_box(index)[vertex]
        self.data.flip_horizontal_shapes(center=center, index=index)
        self.refresh()

    def rotate_shapes(self, angle, vertex=-1, index=True):
        """Perfroms a rotation on selected shapes
        Parameters
        ----------
        angle : float
            angle specifying rotation of shapes in degrees.
        vertex : int
            coordinate of bounding box to use as center of rotation.
        index : bool, list, int
            index of objects to be selected. Where True corresponds to all
            objects, a list of integers to a list of objects, and a single
            integer to that particular object.
        """
        center = self.data.selected_box(index)[vertex]
        self.data.rotate_shapes(angle, center=center, index=index)
        self.refresh()

    def shift_shapes(self, shift, index=True):
        """Perfroms an 2D shift on selected shapes
        Parameters
        ----------
        shift : np.ndarray
            length 2 array specifying shift of shapes.
        index : bool, list, int
            index of objects to be selected. Where True corresponds to all
            objects, a list of integers to a list of objects, and a single
            integer to that particular object.
        """
        self.data.shift_shapes(shift, index=index)
        self.refresh()

    def align_shapes(self, index=True, axis=0, location=1, to_canvas=False):
        """Aligns selected shapes either in horizontal or vertical axis to
        the left, center, right, or top, center, bottom. If to_canvas is True
        then it aligns to the canvas.
        Parameters
        ----------
        index : bool, list, int
            index of objects to be selected. Where True corresponds to all
            objects, a list of integers to a list of objects, and a single
            integer to that particular object.
        axis : int
            integer specifying axis to align along. 0 corresponds to horizontal
            1 corresponds to vertical
        location : int
            location that objects are to be aligned too. One of 0, 1, 2 for
            either left, center, right, or top, center, bottom for Horizontal
            and vertical axes respectively.
        to_canvas : bool
            Bool specifying if to align to canvas or not.
        """
        if to_canvas:
            max_shape = self.viewer.dimensions.max_shape
            box = self.data._expand_box(np.array([[0, 0], max_shape[:2]]))
        else:
            box = self.data.selected_box(index)
        coords = [[7, 8, 3], [1, 8, 5]]
        coord = coords[axis][location]
        align_point = box[coord][axis]

        if index is True:
            index = list(range(len(self.data.id)))

        if type(index) is list:
            for i in index:
                box = self.data.selected_box(i)
                shift = [0, 0]
                shift[axis] = align_point - box[coord][axis]
                self.data.shift_shapes(shift, index=i)
        else:
            box = self.data.selected_box(index)
            shift = [0, 0]
            shift[axis] = align_point - box[coord][axis]
            self.data.shift_shapes(shift, index=index)
        self.refresh()

    def distribute_shapes(self, index=True, axis=0, location=1, to_canvas=False):
        """Distributes selected shapes either in horizontal or vertical axis to
        the left, center, right, or top, center, bottom. If to_canvas is True
        then it dispributes along the canvas.
        Parameters
        ----------
        index : bool, list, int
            index of objects to be selected. Where True corresponds to all
            objects, a list of integers to a list of objects, and a single
            integer to that particular object.
        axis : int
            integer specifying axis to align along. 0 corresponds to horizontal
            1 corresponds to vertical
        location : int
            location that objects are to be aligned too. One of 0, 1, 2 for
            either left, center, right, or top, center, bottom for Horizontal
            and vertical axes respectively.
        to_canvas : bool
            Bool specifying if to align to canvas or not.
        """
        if to_canvas:
            max_shape = self.viewer.dimensions.max_shape
            box = self.data._expand_box(np.array([[0, 0], max_shape[:2]]))
        else:
            box = self.data.selected_box(index)
        coords = [[7, 8, 3], [1, 8, 5]]
        coord = coords[axis][location]
        align_points = [box[coords[axis][0]][axis], box[coords[axis][2]][axis]]

        if index is True:
            index = list(range(len(self.data.id)))

        if type(index) is list:
            num_objects = len(index)
            align_points = np.linspace(align_points[0], align_points[1], num_objects)
            offsets = []
            for obj_ind in index:
                box = self.data.selected_box(obj_ind)
                offsets.append(box[coord][axis])
            offsets = np.array(offsets)
            order = np.argsort(offsets)
            for i in range(len(index)):
                obj_ind = index[order[i]]
                shift = [0, 0]
                shift[axis] = align_points[i] - offsets[order[i]]
                self.data.shift_shapes(shift, index=index[order[i]])
        else:
            align_points = (align_points[0] + align_points[1])/2
            box = self.data.selected_box(index)
            shift = [0, 0]
            shift[axis] = align_points - box[coord][axis]
            self.data.shift_shapes(shift, index=index)
        self.refresh()

    def set_thickness(self, index=True, thickness=1):
        if isinstance(thickness, (list, np.ndarray)):
            if index is True:
                self.data._thickness = thickness
            else:
                self.data._thickness[index] = thickness
        else:
            if index is True:
                self.data._thickness = np.repeat(thickness, len(self.data._thickness))
            else:
                self.data._thickness[index] = np.repeat(thickness, len(index))

        self.data.update_thickness(index)
        self.refresh()

    def move_forward(self, index):
        if type(index) is list:
            z_order = self._z_order.tolist()
            indices = [z_order.index(i) for i in index]
            for i in np.sort(indices):
                self._move_one_forward(z_order[i])
        else:
            self._move_one_forward(index)
        self.z_order = self._z_order

    def move_backward(self, index):
        if type(index) is list:
            z_order = self._z_order.tolist()
            indices = [z_order.index(i) for i in index]
            for i in np.sort(indices)[::-1]:
                self._move_one_backward(z_order[i])
        else:
            self._move_one_backward(index)
        self.z_order = self._z_order

    def move_to_front(self, index):
        if type(index) is list:
            z_order = self._z_order.tolist()
            indices = [z_order.index(i) for i in index]
            for i in np.sort(indices)[::-1]:
                self._move_one_to_front(z_order[i])
        else:
            self._move_one_to_front(index)
        self.z_order = self._z_order

    def move_to_back(self, index):
        if type(index) is list:
            z_order = self._z_order.tolist()
            indices = [z_order.index(i) for i in index]
            for i in np.sort(indices):
                self._move_one_to_back(z_order[i])
        else:
            self._move_one_to_back(index)
        self.z_order = self._z_order

    def _move_one_forward(self, index):
        ind = self._z_order.tolist().index(index)
        if ind != 0:
            self._z_order[ind] = self._z_order[ind-1]
            self._z_order[ind-1] = index

    def _move_one_backward(self, index):
        ind = self._z_order.tolist().index(index)
        if ind != len(self._z_order)-1:
            self._z_order[ind] = self._z_order[ind+1]
            self._z_order[ind+1] = index

    def _move_one_to_front(self, index):
        self._z_order[1:] = self._z_order[self._z_order!=index]
        self._z_order[0] = index

    def _move_one_to_back(self, index):
        self._z_order[:-1] = self._z_order[self._z_order!=index]
        self._z_order[-1] = index

    def hide(self, index=True, object_type=None):
        if index is None:
            self._show_faces = np.array([True for i in range(len(self.data._mesh_faces))])
        else:
            indices = self.data._select_meshes(index=index, meshes=self.data._mesh_faces_index, object_type=object_type)
            self._show_faces[indices] = False
        self.refresh()

    def set_color(self, index=True, edge_color=False, face_color=False):
        if face_color is False:
            pass
        else:
            if type(face_color) is list:
                if index is True:
                    assert(self.data._mesh_faces_index[:, 0].max()<len(face_color))
                    for i in range(len(self.data._mesh_faces_index)):
                        if self.data._mesh_faces_index[i, 2] == 0:
                            self._color_array[i] = Color(face_color[self.data._mesh_faces_index[i, 0]]).rgba
                else:
                    assert(type(index) is list and len(face_color)==len(index))
                    for i in range(len(index)):
                        indices = self.data._select_meshes(index[i], self.data._mesh_faces_index, 0)
                        color = Color(face_color[i]).rgba
                        self._color_array[indices] = color
            else:
                indices = self.data._select_meshes(index, self.data._mesh_faces_index, 0)
                color = Color(face_color).rgba
                self._color_array[indices] = color
        if edge_color is False:
            pass
        else:
            if type(edge_color) is list:
                if index is True:
                    assert(self.data._mesh_faces_index[:, 0].max()<len(edge_color))
                    for i in range(len(self.data._mesh_faces_index)):
                        if self.data._mesh_faces_index[i, 2] == 1:
                            self._color_array[i] = Color(edge_color[self.data._mesh_faces_index[i, 0]]).rgba
                else:
                    assert(type(index) is list and len(edge_color)==len(index))
                    for i in range(len(index)):
                        indices = self.data._select_meshes(index[i], self.data._mesh_faces_index, 1)
                        color = Color(edge_color[i]).rgba
                        self._color_array[indices] = color
            else:
                indices = self.data._select_meshes(index, self.data._mesh_faces_index, 1)
                color = Color(edge_color).rgba
                self._color_array[indices] = color
        self.refresh()

    def _shape_at(self, indices):
        """Determines if any shapes at given indices by looking inside triangle
        meshes.
        Parameters
        ----------
        indices : sequence of int
            Indices to check if shape at.
        """
        # Check if mouse inside vertex of bounding box
        if len(self._selected_shapes) > 0:
            box = self.data.selected_box(self._selected_shapes)[:-1]
            distances = abs(box - indices[:2])

            # Get the vertex sizes
            sizes = self._vertex_size

            # Check if any matching vertices
            matches = np.all(distances <=  self._vertex_size/2, axis=1).nonzero()
            if len(matches[0]) > 0:
                return [self._selected_shapes[0], matches[0][-1]]

        # Check if mouse inside shape
        triangles = self.data._mesh_vertices[self.data._mesh_faces[self._show_faces]]
        shapes = self.data._mesh_faces_index[inside_triangles(triangles - indices[:2])]

        if len(shapes) > 0:
            indices = shapes[:, 0]
            z_list = self._z_order.tolist()
            order_indices = np.array([z_list.index(m) for m in indices])
            ordered_shapes = indices[np.argsort(order_indices)]
            return [ordered_shapes[0], None]
        else:
            return [None, None]

    def _set_view_slice(self, indices):
        """Sets the view given the indices to slice with.
        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.
        """
        show_faces = self._show_faces[self._z_order_faces]
        faces = self.data._mesh_faces[self._z_order_faces][show_faces]
        colors = self._color_array[self._z_order_faces][show_faces]
        vertices = self.data._mesh_vertices
        if len(faces) == 0:
            self._node._subvisuals[3].set_data(vertices=None, faces=None)
        else:
            self._node._subvisuals[3].set_data(vertices=vertices, faces=faces,
                                               face_colors=colors)
        self._need_visual_update = True
        #self._set_highlight()
        self._update()

    def _set_highlight(self):
        if self._highlight and (self._hover_shapes[0] is not None or len(self._selected_shapes)>0):

            # show outlines hover shape or any selected shapes
            if len(self._selected_shapes)>0:
                index = copy(self._selected_shapes)
                if self._hover_shapes[0] is not None:
                    if self._hover_shapes[0] in index:
                        pass
                    else:
                        index.append(self._hover_shapes[0])
                index.sort()
            else:
                index = self._hover_shapes[0]

            faces_indices = self.data._select_meshes(index, self.data._mesh_faces_index, 1)
            vertices_indices = self.data._select_meshes(index, self.data._mesh_vertices_index, 1)
            vertices = self.data._mesh_vertices_centers[vertices_indices] + self._highlight_thickness*self.data._mesh_vertices_offsets[vertices_indices]
            faces = self.data._mesh_faces[faces_indices]
            if type(index) is list:
                faces_index = self.data._mesh_faces_index[faces_indices][:,0]
                starts = np.unique(self.data._mesh_vertices_index[vertices_indices][:,0], return_index=True)[1]
                for i, ind in enumerate(index):
                    faces[faces_index==ind] = faces[faces_index==ind] - vertices_indices[starts[i]] + starts[i]
            else:
                faces = faces - vertices_indices[0]
            self._node._subvisuals[2].set_data(vertices=vertices, faces=faces,
                                               color=self._highlight_color)
        else:
            self._node._subvisuals[2].set_data(vertices=None, faces=None)

        if self._highlight and len(self._selected_shapes) > 0:
            box = self.data.selected_box(self._selected_shapes)[:-1]
            if self._hover_shapes[0] is None:
                face_color = 'white'
            elif self._hover_shapes[1] is None:
                face_color = 'white'
            else:
                face_color = self._highlight_color
            edge_color = self._highlight_color
            self._node._subvisuals[0].set_data(box, size=8, face_color=face_color,
                                               edge_color=edge_color, edge_width=1,
                                               symbol='square', scaling=True)
            self._node._subvisuals[1].set_data(pos=box[[0, 2, 4, 6, 0]],
                                               color=edge_color, width=1)
        else:
            self._node._subvisuals[0].set_data(np.empty((0, 2)), size=0)
            self._node._subvisuals[1].set_data(pos=None, width=0)

    def _get_coord(self, position, indices):
        max_shape = self.viewer.dimensions.max_shape
        transform = self.viewer._canvas.scene.node_transform(self._node)
        pos = transform.map(position)
        pos = [pos[1], pos[0]]
        coord = copy(indices)
        coord[0] = pos[1]
        coord[1] = pos[0]
        return np.array(coord)

    def get_value(self, position, indices):
        """Returns coordinates, values, and a string
        for a given mouse position and set of indices.
        Parameters
        ----------
        position : sequence of two int
            Position of mouse cursor in canvas.
        indices : sequence of int or slice
            Indices that make up the slice.
        Returns
        ----------
        coord : sequence of int
            Position of mouse cursor in data.
        value : int or float or sequence of int or float
            Value of the data at the coord.
        msg : string
            String containing a message that can be used as
            a status update.
        """
        coord = self._get_coord(position, indices)
        value = self._shape_at(coord)
        coord_shift = [int(coord[1]), int(coord[0])]
        msg = f'{coord_shift}'
        if value[0] is not None:
            msg = msg + ', ' + self.name + ', index ' + str(value[0])
        return coord, value[0], msg

    # def _add(self, coord, br=None):
    #     """Returns coordinates, values, and a string
    #     for a given mouse position and set of indices.
    #     Parameters
    #     ----------
    #     position : sequence of two int
    #         Position of mouse cursor in canvas.
    #     indices : sequence of int or slice
    #         Indices that make up the slice.
    #     """
    #     max_shape = self.viewer.dimensions.max_shape
    #
    #     if br is None:
    #         tl = [coord[0]-25, coord[1]-25, *coord[2:]]
    #         br = [coord[0]+25, coord[1]+25, *coord[2:]]
    #         index = None
    #     else:
    #         tl = coord
    #         br = br
    #         if br[0] == tl[0]:
    #             br[0] = tl[0]+1
    #         if br[1] == tl[1]:
    #             br[1] = tl[1]+1
    #         index = 2
    #
    #     if br[0] > max_shape[0]-1:
    #         br[0] = max_shape[0]-1
    #         tl[0] = max_shape[0]-1-50
    #     if br[1] > max_shape[1]-1:
    #         br[1] = max_shape[1]-1
    #         tl[1] = max_shape[1]-1-50
    #     if tl[0] < 0:
    #         br[0] = 50
    #         tl[0] = 0
    #     if tl[1] < 0:
    #         br[1] = 50
    #         tl[1] = 0
    #
    #     # print('to_add', [[tl, br]])
    #     # print('data', self.data)
    #     # print('index', index)
    #     self.data = append(self.data, [[tl, br]], axis=0)
    #     self._selected_shapes = [len(self.data)-1, index]
    #     self.refresh()
    #
    # def _remove(self, coord):
    #     """Returns coordinates, values, and a string
    #     for a given mouse position and set of indices.
    #     Parameters
    #     ----------
    #     position : sequence of two int
    #         Position of mouse cursor in canvas.
    #     indices : sequence of int or slice
    #         Indices that make up the slice.
    #     """
    #     index = self._selected_shapes
    #     if index is None:
    #         pass
    #     else:
    #         self.data = delete(self.data, index[0], axis=0)
    #         self._selected_shapes = self._get_selected_shapes(coord)
    #         self.refresh()
    #
    def _move(self, coord):
        """Moves object at given mouse position
        and set of indices.
        Parameters
        ----------
        coord : sequence of two int
            Position of mouse cursor in data.
        """
        self._is_moving=True
        index = self._selected_shapes
        vertex = self._selected_vertex
        if len(index) > 0:
            if vertex is None:
                #Check where dragging box from to move whole object
                if self._drag_start is None:
                    center = self.data.selected_box(index)[-1]
                    self._drag_start = coord - center
                center = self.data.selected_box(index)[-1]
                shift = coord - center - self._drag_start
                self.shift_shapes(shift, index=index)
                self._set_highlight()
            else:
                #Vertex is being dragged so resize object
                box = self.data.selected_box(index)
                if self._fixed_vertex is None:
                    self._fixed_index = np.mod(vertex+4,8)
                    self._fixed_vertex = box[self._fixed_index]
                    self._aspect_ratio = (box[4][1]-box[0][1])/(box[4][0]-box[0][0])

                size = box[np.mod(self._fixed_index+4,8)] - box[self._fixed_index]

                if np.mod(self._fixed_index, 2) == 0:
                    # corner selected
                    fixed = self._fixed_vertex
                    new = coord
                    if self._fixed_aspect:
                        ratio = abs((new - fixed)[1]/(new - fixed)[0])
                        if ratio>self._aspect_ratio:
                            new[1] = fixed[1]+(new[1]-fixed[1])*self._aspect_ratio/ratio
                        else:
                            new[0] = fixed[0]+(new[0]-fixed[0])*ratio/self._aspect_ratio
                    scale = (new - fixed)/size
                elif np.mod(self._fixed_index, 4) == 1:
                    # top selected
                    fixed = self._fixed_vertex[1]
                    new = coord[1]
                    scale = np.array([1, (new - fixed)/size[1]])
                else:
                    # side selected
                    fixed = self._fixed_vertex[0]
                    new = coord[0]
                    scale = np.array([(new - fixed)/size[0], 1])

                # prvent box from dissappearing if shrunk near 0
                scale[scale==0]=1
                #print(self._fixed_index, scale)
                self.scale_shapes(scale, vertex=self._fixed_index, index=index)
                self._set_highlight()

    def _select(self):
        if (self._selected_shapes == self._selected_shapes_stored and
            self._hover_shapes == self._hover_shapes_stored):
            return
        self._highlight = True
        self._selected_shapes_stored = copy(self._selected_shapes)
        self._hover_shapes_stored = copy(self._hover_shapes)
        self._set_highlight()

    def _unselect(self):
        if self._highlight:
            self._highlight = False
            self._selected_shapes_stored = []
            self._hover_shapes_stored = [None, None]
            self._set_highlight()

    def interact(self, position, indices, mode=True, dragging=False, shift=False, ctrl=False,
        pressed=False, released=False, moving=False):
        """Highlights object at given mouse position
        and set of indices.
        Parameters
        ----------
        position : sequence of two int
            Position of mouse cursor in canvas.
        indices : sequence of int or slice
            Indices that make up the slice.
        """
        # Adjust shape of box on shift key press if currently moving
        if not self._fixed_aspect == shift:
            self._fixed_aspect = shift
            if self._is_moving:
                coord = self._get_coord(position, indices)
                self._move(coord)

        if mode is None:
            #If not in edit or addition mode unselect all
            self._selected_shapes = []
            self._unselect()
        elif mode == 'edit':
            #If in edit mode
            coord = self._get_coord(position, indices)
            if pressed:
                #print('layer press', self._is_moving)
                if not self._is_moving:
                    shape = self._shape_at(coord)
                    self._selected_vertex = shape[1]
                    if shift and shape[0] is not None:
                        if shape[0] in self._selected_shapes:
                            self._selected_shapes.remove(shape[0])
                        else:
                            self._selected_shapes.append(shape[0])
                    elif shape[0] is not None:
                        if shape[0] not in self._selected_shapes:
                            self._selected_shapes = [shape[0]]
                    else:
                        self._selected_shapes = []
                    self._select()
            elif moving and dragging:
                #print('layer moving!!!!')
                #Drag any selected shapes
                self._move(coord)
            elif released:
                #print('layer release!!!!')
                shape = self._shape_at(coord)
                if not self._is_moving:
                    if shift and shape[0] is not None:
                        pass
                    elif shape[0] is not None:
                        self._selected_shapes = [shape[0]]
                    elif not shift:
                        self._selected_shapes = []
                self._is_moving=False
                self._drag_start=None
                self._fixed_vertex = None
                self._selected_vertex = None
                self._hover_shapes = shape
                self._select()
            elif self._is_moving:
                #print('moving passsss!!!!')
                pass
            else:
                #Highlight boxes if over any
                self._hover_shapes = self._shape_at(coord)
                self._select()
        elif mode == 'add':
            self._selected_shapes = []
            self._unselect()
        #     #If in addition mode
        #     coord = self._get_coord(position, indices)
        #     if pressed:
        #         #Start add a new box
        #         self._ready_to_create_box = True
        #         self._creating_box = False
        #         self._create_tl = coord
        #     elif moving and dragging:
        #         #If moving and dragging check if ready to make new box
        #         if self._ready_to_create_box:
        #             self.highlight = True
        #             self._add(self._create_tl, coord)
        #             self._ready_to_create_box = False
        #             self._creating_box = True
        #         elif self._creating_box:
        #             #If making a new box, update it's position
        #             self._move(coord)
        #     elif released and dragging:
        #         #One release add new box
        #         if self._creating_box:
        #             self._creating_box = False
        #             self._unselect()
        #             self._fixed = None
        #         else:
        #             self._add(coord)
        #             self._ready_to_create_box = False
        #         self._is_moving=False
        #     elif released:
        #         self._is_moving=False
        #     else:
        #         self._unselect()
        else:
            pass
