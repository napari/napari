import numpy as np
from copy import copy
from .shape import Shape

class ShapesList():
    """List of shapes class.
    Parameters
    ----------
    shapes : list
        List of dictionaries, where each dictionary corresponds to one shape
        and has the keys corresponding to the keyword args of the Shapes class.
        `shape_type` and `data` are required. `edge_width`, `edge_color`,
        `face_color`, and `z_order` are optional.
    """
    _mesh_types = ['face', 'edge']

    def __init__(self, shapes):

        self.shapes = []
        self._vertices = np.empty((0, 2)) # Array of M vertices from all N shapes
        self._index = np.empty((0), dtype=int) # Shape index (0, ..., N-1) for each of M vertices
        self._z_order = np.empty((0), dtype=int) # Length N array of z_order of each shape

        self._mesh_vertices = np.empty((0, 2)) # Mx2 array of vertices of triangles
        self._mesh_vertices_index = np.empty((0, 2), dtype=int) #Mx2 array of shape index and types of vertex (face / edge)
        self._mesh_triangles = np.empty((0, 3), dtype=np.uint32) # Px3 array of vertex indices that form a triangle
        self._mesh_triangles_index = np.empty((0, 2), dtype=int) #Px2 array of shape index and types of triangle (face / edge)
        self._mesh_triangles_colors = np.empty((0, 4)) #Px4 array of rgba mesh triangle colors
        self._mesh_triangles_z_order = np.empty((0), dtype=int) #Length P array of mesh triangle z_order

        for s in shapes:
            self.add_shape(s)

    def add_shape(self, shape, shape_index=None):
        """Adds a single dictionary representing a shape object or a Shape
        Parameters
        ----------
        shape_dictionary : Shape | dict
            Either an instance of the Shape class or a dictionary that has the
            keyword arguments corresponding to the Shape class constructor. In
            this case, `shape_type` and `data` are required and `edge_width`,
            `edge_color`, `face_color`, and `z_order` are optional.
        shape_index : None | int
            If None, then index of shape if the equal to the number of shapes
            in the list
        """
        if type(shape) is dict:
            shape = Shape(**shape)

        if shape_index is None:
            shape_index = len(self.shapes)
            self.shapes.append(shape)
            self._z_order = np.append(self._z_order, shape.z_order)
        else:
            self.shapes[shape_index] = shape
            self._z_order[shape_index] = shape.z_order

        self._vertices = np.append(self._vertices, shape.data, axis=0)
        self._index = np.append(self._index, np.repeat(shape_index, len(shape.data)), axis=0)

        # Add edges to mesh
        m = len(self._mesh_vertices)
        vertices = shape._edge_vertices + shape.edge_width*shape._edge_offsets
        self._mesh_vertices = np.append(self._mesh_vertices, vertices, axis=0)
        index = np.repeat([[shape_index, 1]], len(vertices), axis=0)
        self._mesh_vertices_index = np.append(self._mesh_vertices_index, index, axis=0)

        triangles = shape._edge_triangles + m
        self._mesh_triangles = np.append(self._mesh_triangles, triangles, axis=0)
        index = np.repeat([[shape_index, 1]], len(triangles), axis=0)
        self._mesh_triangles_index = np.append(self._mesh_triangles_index, index, axis=0)
        color = np.repeat([shape.edge_color.rgba], len(triangles), axis=0)
        self._mesh_triangles_colors = np.append(self._mesh_triangles_colors, color, axis=0)
        # Need to fix z_order here!!!!!!
        order = np.repeat(shape.z_order, len(triangles))
        self._mesh_triangles_z_order = np.append(self._mesh_triangles_z_order, order, axis=0)

        # Add faces to mesh
        m = len(self._mesh_vertices)
        vertices = shape._face_vertices
        self._mesh_vertices = np.append(self._mesh_vertices, vertices, axis=0)
        index = np.repeat([[shape_index, 0]], len(vertices), axis=0)
        self._mesh_vertices_index = np.append(self._mesh_vertices_index, index, axis=0)

        triangles = shape._face_triangles + m
        self._mesh_triangles = np.append(self._mesh_triangles, triangles, axis=0)
        index = np.repeat([[shape_index, 0]], len(triangles), axis=0)
        self._mesh_triangles_index = np.append(self._mesh_triangles_index, index, axis=0)
        color = np.repeat([shape.face_color.rgba], len(triangles), axis=0)
        self._mesh_triangles_colors = np.append(self._mesh_triangles_colors, color, axis=0)
        # Need to fix z_order here!!!!!!
        order = np.repeat(shape.z_order, len(triangles))
        self._mesh_triangles_z_order = np.append(self._mesh_triangles_z_order, order, axis=0)

    def set_shapes(self, shapes):
        """Removes all shapes and then adds in the new ones
        Parameters
        ----------
        shapes : list
            List of dictionaries, where each dictionary corresponds to one shape
            and has the keys corresponding to the keyword args of the Shapes class.
            `shape_type` and `data` are required. `edge_width`, `edge_color`,
            `face_color`, and `z_order` are optional.
        """
        self.remove_all_shapes()
        for s in shapes:
            self.add_shape(s)

    def remove_all_shapes(self):
        """Removes all shapes
        """
        self.shapes = []
        self._vertices = np.empty((0, 2))
        self._index = np.empty((0), dtype=int)
        self._z_order = np.empty((0), dtype=int)

        self._mesh_vertices = np.empty((0, 2))
        self._mesh_vertices_index = np.empty((0, 2), dtype=int)
        self._mesh_triangles = np.empty((0, 3), dtype=np.uint32)
        self._mesh_triangles_index = np.empty((0, 2), dtype=int)
        self._mesh_triangles_colors = np.empty((0, 4))
        self._mesh_triangles_z_order = np.empty((0), dtype=int)

    def remove_one_shape(self, index, renumber=True):
        """Removes a single shape located at index.
        Parameters
        ----------
        index : int
            Location in list of the shape to be removed.
        renumber : bool
            Bool to indicate whether to renumber all shapes or not. If not the
            expectation is that this shape is being immediately readded to the
            list using `add_shape`.
        """
        self._vertices = self._vertices[self._index!=index]
        self._index = self._index[self._index!=index]

        # Remove triangles
        indices = self._mesh_triangles_index[:, 0] == index
        self._mesh_triangles = np.delete(self._mesh_triangles, indices, axis=0)
        self._mesh_triangles_colors = np.delete(self._mesh_triangles_colors, indices, axis=0)
        # Need to fix z_order here!!!!!!
        self._mesh_triangles_z_order = np.delete(self._mesh_triangles_z_order, indices, axis=0)
        self._mesh_triangles_index = np.delete(self._mesh_triangles_index, indices, axis=0)

        # Remove vertices
        indices = self._mesh_vertices_index[:, 0] == index
        self._mesh_vertices_index = np.delete(self._mesh_vertices_index, indices, axis=0)
        self._mesh_vertices = np.delete(self._mesh_vertices, indices, axis=0)
        self._mesh_triangles[self._mesh_triangles>indices[0]] = self._mesh_triangles[self._mesh_triangles>indices[0]] - len(indices)

        if renumber:
            del self.shapes[index]
            del self._z_order[index]
            self._index[self._index>index] = self._index[self._index>index]-1
            self._mesh_triangles_index[self._mesh_triangles_index[:,0]>index,0] = self._mesh_triangles_index[self._mesh_triangles_index[:,0]>index,0]-1
            self._mesh_vertices_index[self._mesh_vertices_index[:,0]>index,0] = self._mesh_vertices_index[self._mesh_vertices_index[:,0]>index,0]-1

    def _update_mesh_vertices(self, index, edge=False, face=False):
        """Updates the mesh vertex data for a single shape located at index.
        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        edge : bool
            Bool to indicate whether to update mesh vertices corresponding to edges
        face : bool
            Bool to indicate whether to update mesh vertices corresponding to faces
        """
        if edge:
            indices = np.all(self._mesh_vertices_index == [index, 1], axis=1)
            self._mesh_vertices[indices] = (self.shapes[index]._edge_vertices +
            self.shapes[index].edge_width*self.shapes[index]._edge_offsets)

        if face:
            indices = np.all(self._mesh_vertices_index == [index, 0], axis=1)
            self._mesh_vertices[indices] = self.shapes[index]._face_vertices

    def edit(self, index, data):
        """Updates the z order of a single shape located at index.
        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        data : np.ndarray
            Nx2 array of vertices.
        """
        self.shapes[index].data = data
        shape = self.shapes[index]
        self.remove_one_shape(index, renumber=False)
        self.add_shape(shape)

    def update_edge_width(self, index, edge_width):
        """Updates the edge width of a single shape located at index.
        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        edge_width : float
            thickness of lines and edges.
        """
        self.shapes[index].edge_width = edge_width
        self._update_mesh_vertices(index, edge=True)

    def update_edge_color(self, index, edge_color):
        """Updates the edge color of a single shape located at index.
        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        edge_color : str | tuple
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3 or
            4 elements.
        """
        self.shapes[index].edge_color = edge_color
        indices = np.all(self._mesh_triangles_index == [index, 1], axis=1)
        self._mesh_triangles_colors[indices] = self.shapes[index].edge_color.rgba

    def update_face_color(self, index, face_color):
        """Updates the face color of a single shape located at index.
        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        face_color : str | tuple
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3 or
            4 elements.
        """
        self.shapes[index].face_color = face_color
        indices = np.all(self._mesh_triangles_index == [index, 0], axis=1)
        self._mesh_triangles_colors[indices] = self.shapes[index].face_color.rgba

    def update_z_order(self, index, z_order):
        """Updates the z order of a single shape located at index.
        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        z_order : int
            Specifier of z order priority. Shapes with higher z order are displayed
            ontop of others.
        """
        self.shapes[index].z_order = z_order
        indices = self._mesh_triangles_index[:, 0] == index
        self._mesh_triangles_z_order[indices] = self.shapes[index].z_order

    def shift(self, index, shift):
        """Perfroms a 2D shift on a single shape located at index
        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        shift : np.ndarray
            length 2 array specifying shift of shapes.
        """
        self.shapes[index].shift(shift)
        self._update_mesh_vertices(index, edge=True, face=True)

    def scale(self, index, scale, center=None):
        """Perfroms a scaling on a single shape located at index
        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        scale : float, list
            scalar or list specifying rescaling of shape.
        center : list
            length 2 list specifying coordinate of center of scaling.
        """
        self.shapes[index].scale(scale, center=center)
        self._update_mesh_vertices(index, edge=True, face=True)

    def rotate(self, index, angle, center=None):
        """Perfroms a rotation on a single shape located at index
        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        angle : float
            angle specifying rotation of shape in degrees.
        center : list
            length 2 list specifying coordinate of center of rotation.
        """
        self.shapes[index].rotate(angle, center=center)
        self._update_mesh_vertices(index, edge=True, face=True)

    def flip(self, index, axis, center=None):
        """Perfroms an vertical flip on a single shape located at index
        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        axis : int
            integer specifying axis of flip. `0` flips horizontal, `1` flips
            vertical.
        center : list
            length 2 list specifying coordinate of center of flip axes.
        """
        self.shapes[index].flip(axis, center=center)
        self._update_mesh_vertices(index, edge=True, face=True)
