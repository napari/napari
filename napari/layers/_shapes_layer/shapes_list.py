import numpy as np
from copy import copy
from .shape import Shape

class ShapesList():
    """List of shapes class.
    Parameters
    ----------
    data : np.array | list | Shape
        List of Shape objects of list of np.array of data or np.array. Each
        element of the list (or now of the np.array) corresponds to one shape.
        If a list of Shape objects is passed the other shape specific keyword
        arguments are ignored.
    shape_type : string | list
        String of shape shape_type, must be one of "{'line', 'rectangle', 'ellipse',
        'path', 'polygon'}". If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape otherwise
        the same value will be used for all shapes.
    edge_width : float | list
        thickness of lines and edges. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape otherwise
        the same value will be used for all shapes.
    edge_color : str | tuple | list
        If string can be any color name recognized by vispy or hex value if
        starting with `#`. If array-like must be 1-dimensional array with 3 or
        4 elements. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape otherwise
        the same value will be used for all shapes.
    face_color : str | tuple | list
        If string can be any color name recognized by vispy or hex value if
        starting with `#`. If array-like must be 1-dimensional array with 3 or
        4 elements. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape otherwise
        the same value will be used for all shapes.
    opacity : float | list
        Opacity of the shape, must be between 0 and 1. If a list is supplied it
        must be the same length as the length of `data` and each element will
        be applied to each shape otherwise the same value will be used for all shapes.
    z_index : int | list
        Specifier of z order priority. Shapes with higher z order are displayed
        ontop of others. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape otherwise
        the same value will be used for all shapes.
    """
    _mesh_types = ['face', 'edge']

    def __init__(self, data, shape_type='rectangle', edge_width=1, edge_color='black',
                 face_color='white', opacity=1, z_index=0):

        self.shapes = []
        self._vertices = np.empty((0, 2)) # Array of M vertices from all N shapes
        self._index = np.empty((0), dtype=int) # Shape index (0, ..., N-1) for each of M vertices
        self._z_index = np.empty((0), dtype=int) # Length N array of z_index of each shape

        self._mesh_vertices = np.empty((0, 2)) # Mx2 array of vertices of triangles
        self._mesh_vertices_centers = np.empty((0, 2)) # Mx2 array of vertices of triangles
        self._mesh_vertices_offsets = np.empty((0, 2)) # Mx2 array of vertices of triangles
        self._mesh_vertices_index = np.empty((0, 2), dtype=int) #Mx2 array of shape index and types of vertex (face / edge)
        self._mesh_triangles = np.empty((0, 3), dtype=np.uint32) # Px3 array of vertex indices that form a triangle
        self._mesh_triangles_index = np.empty((0, 2), dtype=int) #Px2 array of shape index and types of triangle (face / edge)
        self._mesh_triangles_colors = np.empty((0, 4)) #Px4 array of rgba mesh triangle colors
        self._mesh_triangles_z_index = np.empty((0), dtype=int) #Length P array of mesh triangle z_index

        if type(data) is Shape:
            # If a single shape has been passed
            self.add(data)
        elif len(data) > 0:
            if type(data[0]) is Shape:
                # If list of shapes has been passed
                for d in data:
                    self.add(d)
            elif np.array(data[0]).ndim == 1:
                # If a single array for a shape has been passed
                self.add(data, shape_type=shape_type, edge_width=edge_width,
                         edge_color=edge_color, face_color=face_color,
                         opacity=opacity, z_index=z_index)
            else:
                # If list of arrays has been passed
                for i, d in enumerate(data):
                    if type(shape_type) is list or type(shape_type) is np.ndarray:
                        st = shape_type[i]
                    else:
                        st = shape_type
                    if type(edge_width) is list or type(edge_width) is np.ndarray:
                        ew = edge_width[i]
                    else:
                        ew = edge_width
                    if type(edge_color) is list or type(edge_color) is np.ndarray:
                        if np.isscalar(edge_color[i]):
                            ec = edge_color
                        else:
                            ec = edge_color[i]
                    else:
                        ec = edge_color
                    if type(face_color) is list or type(face_color) is np.ndarray:
                        if np.isscalar(face_color[i]):
                            fc = face_color
                        else:
                            fc = face_color[i]
                    else:
                        fc = face_color
                    if type(z_index) is list or type(z_index) is np.ndarray:
                        z = z_index[i]
                    else:
                        z = z_index
                    if type(opacity) is list or type(opacity) is np.ndarray:
                        o = opacity[i]
                    else:
                        o = opacity
                    self.add(d, shape_type=st, edge_width=ew, edge_color=ec,
                             face_color=fc, opacity=o, z_index=z)

    def add(self, data, shape_type='rectangle', edge_width=1, edge_color='black',
                 face_color='white', opacity=1, z_index=0, shape_index=None):
        """Adds a single Shape object
        Parameters
        ----------
        data : np.ndarray | Shape
            Nx2 array of vertices or instance of the Shape class. If a Shape is passed
            the other parameters are ignored.
        shape_type : string
            String of shape shape_type, must be one of "{'line', 'rectangle', 'ellipse',
            'path', 'polygon'}".
        edge_width : float
            thickness of lines and edges.
        edge_color : str | tuple
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3 or
            4 elements.
        face_color : str | tuple
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3 or
            4 elements.
        opacity : float
            Opacity of the shape, must be between 0 and 1.
        z_index : int
            Specifier of z order priority. Shapes with higher z order are displayed
            ontop of others.
        shape : Shape
            An instance of the Shape class
        shape_index : None | int
            If int then edits the shape date at current index. To be used in
            conjunction with `remove` when renumber is `False`. If None, then
            appends a new shape to end of shapes list
        """
        if type(data) is Shape:
            shape = data
        else:
            shape = Shape(data, shape_type=shape_type, edge_width=edge_width,
                         edge_color=edge_color, face_color=face_color,
                         opacity=opacity, z_index=z_index)

        if shape_index is None:
            shape_index = len(self.shapes)
            self.shapes.append(shape)
            self._z_index = np.append(self._z_index, shape.z_index)
        else:
            self.shapes[shape_index] = shape
            self._z_index[shape_index] = shape.z_index

        self._vertices = np.append(self._vertices, shape.data, axis=0)
        self._index = np.append(self._index, np.repeat(shape_index, len(shape.data)), axis=0)

        # Add faces to mesh
        m = len(self._mesh_vertices)
        vertices = shape._face_vertices
        self._mesh_vertices = np.append(self._mesh_vertices, vertices, axis=0)
        vertices = shape._face_vertices
        self._mesh_vertices_centers = np.append(self._mesh_vertices_centers, vertices, axis=0)
        vertices = np.zeros(shape._face_vertices.shape)
        self._mesh_vertices_offsets = np.append(self._mesh_vertices_offsets, vertices, axis=0)
        index = np.repeat([[shape_index, 0]], len(vertices), axis=0)
        self._mesh_vertices_index = np.append(self._mesh_vertices_index, index, axis=0)

        triangles = shape._face_triangles + m
        self._mesh_triangles = np.append(self._mesh_triangles, triangles, axis=0)
        index = np.repeat([[shape_index, 0]], len(triangles), axis=0)
        self._mesh_triangles_index = np.append(self._mesh_triangles_index, index, axis=0)
        color = shape.face_color.rgba
        color[3] = color[3]*shape.opacity
        color_array = np.repeat([color], len(triangles), axis=0)
        self._mesh_triangles_colors = np.append(self._mesh_triangles_colors, color_array, axis=0)
        # Need to fix z_index here!!!!!!
        order = np.repeat(shape.z_index, len(triangles))
        self._mesh_triangles_z_index = np.append(self._mesh_triangles_z_index, order, axis=0)

        # Add edges to mesh
        m = len(self._mesh_vertices)
        vertices = shape._edge_vertices + shape.edge_width*shape._edge_offsets
        self._mesh_vertices = np.append(self._mesh_vertices, vertices, axis=0)
        vertices = shape._edge_vertices
        self._mesh_vertices_centers = np.append(self._mesh_vertices_centers, vertices, axis=0)
        vertices = shape._edge_offsets
        self._mesh_vertices_offsets = np.append(self._mesh_vertices_offsets, vertices, axis=0)
        index = np.repeat([[shape_index, 1]], len(vertices), axis=0)
        self._mesh_vertices_index = np.append(self._mesh_vertices_index, index, axis=0)

        triangles = shape._edge_triangles + m
        self._mesh_triangles = np.append(self._mesh_triangles, triangles, axis=0)
        index = np.repeat([[shape_index, 1]], len(triangles), axis=0)
        self._mesh_triangles_index = np.append(self._mesh_triangles_index, index, axis=0)
        color = shape.edge_color.rgba
        color[3] = color[3]*shape.opacity
        color_array = np.repeat([color], len(triangles), axis=0)
        self._mesh_triangles_colors = np.append(self._mesh_triangles_colors, color_array, axis=0)
        # Need to fix z_index here!!!!!!
        order = np.repeat(shape.z_index, len(triangles))
        self._mesh_triangles_z_index = np.append(self._mesh_triangles_z_index, order, axis=0)
        
    def remove_all(self):
        """Removes all shapes
        """
        self.shapes = []
        self._vertices = np.empty((0, 2))
        self._index = np.empty((0), dtype=int)
        self._z_index = np.empty((0), dtype=int)

        self._mesh_vertices = np.empty((0, 2))
        self._mesh_vertices_centers = np.empty((0, 2))
        self._mesh_vertices_offsets = np.empty((0, 2))
        self._mesh_vertices_index = np.empty((0, 2), dtype=int)
        self._mesh_triangles = np.empty((0, 3), dtype=np.uint32)
        self._mesh_triangles_index = np.empty((0, 2), dtype=int)
        self._mesh_triangles_colors = np.empty((0, 4))
        self._mesh_triangles_z_index = np.empty((0), dtype=int)

    def remove(self, index, renumber=True):
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
        indices = self._index != index
        self._vertices = self._vertices[indices]
        self._index = self._index[indices]

        # Remove triangles
        indices = self._mesh_triangles_index[:, 0] != index
        self._mesh_triangles = self._mesh_triangles[indices]
        self._mesh_triangles_colors = self._mesh_triangles_colors[indices]
        # Need to fix z_index here!!!!!!
        self._mesh_triangles_z_index = self._mesh_triangles_z_index[indices]
        self._mesh_triangles_index = self._mesh_triangles_index[indices]

        # Remove vertices
        indices = self._mesh_vertices_index[:, 0] != index
        self._mesh_vertices = self._mesh_vertices[indices]
        self._mesh_vertices_centers = self._mesh_vertices_centers[indices]
        self._mesh_vertices_offsets = self._mesh_vertices_offsets[indices]
        self._mesh_vertices_index = self._mesh_vertices_index[indices]
        indices = np.where(np.invert(indices))[0]
        self._mesh_triangles[self._mesh_triangles>indices[0]] = self._mesh_triangles[self._mesh_triangles>indices[0]] - len(indices)

        if renumber:
            del self.shapes[index]
            self._z_index = np.delete(self._z_index, index)
            self._index[self._index>index] = self._index[self._index>index]-1
            self._mesh_triangles_index[self._mesh_triangles_index[:,0]>index,0] = self._mesh_triangles_index[self._mesh_triangles_index[:,0]>index,0]-1
            self._mesh_vertices_index[self._mesh_vertices_index[:,0]>index,0] = self._mesh_vertices_index[self._mesh_vertices_index[:,0]>index,0]-1

    def _update_mesh_vertices(self, index, edge=False, face=False):
        """Updates the mesh vertex data and vertex data for a single shape
        located at index.
        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        edge : bool
            Bool to indicate whether to update mesh vertices corresponding to edges
        face : bool
            Bool to indicate whether to update mesh vertices corresponding to faces
            and to update the underlying shape vertices
        """
        if edge:
            indices = np.all(self._mesh_vertices_index == [index, 1], axis=1)
            self._mesh_vertices[indices] = (self.shapes[index]._edge_vertices +
                self.shapes[index].edge_width*self.shapes[index]._edge_offsets)
            self._mesh_vertices_centers[indices] = self.shapes[index]._edge_vertices
            self._mesh_vertices_offsets[indices] = self.shapes[index]._edge_offsets

        if face:
            indices = np.all(self._mesh_vertices_index == [index, 0], axis=1)
            self._mesh_vertices[indices] = self.shapes[index]._face_vertices
            self._mesh_vertices_centers[indices] = self.shapes[index]._face_vertices
            indices = self._index == index
            self._vertices[indices] = self.shapes[index].data

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
        self.remove(index, renumber=False)
        self.add(shape, shape_index=index)

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
        color = self.shapes[index].edge_color.rgba
        color[3] = color[3]*self.shapes[index].opacity
        self._mesh_triangles_colors[indices] = color

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
        color = self.shapes[index].face_color.rgba
        color[3] = color[3]*self.shapes[index].opacity
        self._mesh_triangles_colors[indices] = color

    def update_opacity(self, index, opacity):
        """Updates the face color of a single shape located at index.
        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        opacity : float
            Opacity, must be between 0 and 1
        """
        self.shapes[index].opacity = opacity
        indices = np.all(self._mesh_triangles_index == [index, 1], axis=1)
        color = self.shapes[index].edge_color.rgba
        self._mesh_triangles_colors[indices, 3] = color[3]*opacity

        indices = np.all(self._mesh_triangles_index == [index, 0], axis=1)
        color = self.shapes[index].face_color.rgba
        self._mesh_triangles_colors[indices, 3] = color[3]*opacity

    def update_z_index(self, index, z_index):
        """Updates the z order of a single shape located at index.
        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        z_index : int
            Specifier of z order priority. Shapes with higher z order are displayed
            ontop of others.
        """
        self.shapes[index].z_index = z_index
        indices = self._mesh_triangles_index[:, 0] == index
        self._mesh_triangles_z_index[indices] = self.shapes[index].z_index

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

    def transform(self, index, transform):
        """Perfroms a linear transform on a single shape located at index
        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        transform : np.ndarray
            2x2 array specifying linear transform.
        """
        self.shapes[index].transform(transform)
        self._update_mesh_vertices(index, edge=True, face=True)

    def to_list(self, shape_type=None):
        if shape_type is None:
            data = [s.data for s in self.shapes]
        elif shape_type not in Shape._shape_types:
            raise ValueError("""shape_type not recognized, must be one of
                         "{'line', 'rectangle', 'ellipse', 'path',
                         'polygon'}"
                         """)
        else:
            data = [s.data for s in self.shapes if s.shape_type == shapes_type]
        return data
