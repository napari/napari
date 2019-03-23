import numpy as np
from .shape import Shape, Rectangle, Ellipse, Line, Path, Polygon


class ShapeList():
    """List of shapes class.

    Parameters
    ----------
    data : np.ndarray | list | Shape
        List of Shape objects of list of np.ndarray of data or np.ndarray. Each
        element of the list (or now of the np.ndarray) corresponds to one
        shape. If a list of Shape objects is passed the other shape specific
        keyword arguments are ignored.
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
        Opacity of the shapes, must be between 0 and 1. If a list is supplied
        it must be the same length as the length of `data` and each element
        will be applied to each shape otherwise the same value will be used for
        all shapes.
    z_index : int | list
        Specifier of z order priority. Shapes with higher z order are displayed
        ontop of others. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each shape
        otherwise the same value will be used for all shapes.

    Attributes
    ----------
    shapes : list
        Length N list of N shape objects
    _mesh_types : list
        Length two list of the different mesh types corresponding to faces and
        edges
    _vertices : np.ndarray
        Mx2 array of all vertices from all shapes
    _index : np.ndarray
        Length M array with the index (0, ..., N-1) of each shape that each
        vertex corresponds to
    _z_index : np.ndarray
        Length N array with z_index of each shape
    _z_order : np.ndarray
        Length N array with z_order of each shape. This must be a permutation
        of (0, ..., N-1).
    _mesh_vertices : np.ndarray
         Qx2 array of vertices of all triangles for shapes including edges and
         faces
    _mesh_vertices_centers : np.ndarray
         Qx2 array of centers of vertices of triangles for shapes. For vertices
         corresponding to faces these are the same as the actual vertices. For
         vertices corresponding to edges these values should be added to a
         scaled `_mesh_vertices_offsets` to get the actual vertex positions.
         The scaling corresponds to the width of the edge
    _mesh_vertices_offsets : np.ndarray
         Qx2 array of offsets of vertices of triangles for shapes. For vertices
         corresponding to faces these are 0. For vertices corresponding to
         edges these values should be scaled and added to the
         `_mesh_vertices_centers` to get the actual vertex positions.
         The scaling corresponds to the width of the edge
    _mesh_vertices_index : np.ndarray
         Qx2 array of the index (0, ..., N-1) of each shape that each vertex
         corresponds and the mesh type (0, 1) for face or edge.
    _mesh_triangles : np.ndarray
        Px3 array of vertex indices that form the mesh triangles
    _mesh_triangles_index : np.ndarray
        Px2 array of  the index (0, ..., N-1) of each shape that each triangle
        corresponds and the mesh type (0, 1) for face or edge.
    _mesh_triangles_colors : np.ndarray
        Px4 array of the rgba color of each triangle
    _mesh_triangles_z_order : np.ndarray
        Length P array of the z order of each triangle. Must be a permutation
        of (0, ..., P-1)
    _shape_types : dict
        Dictionary of supported shape types and their corresponding objects.
    """
    _mesh_types = ['face', 'edge']

    _shape_types = ({'rectangle': Rectangle, 'ellipse': Ellipse, 'line': Line,
                    'path': Path, 'polygon': Polygon})

    def __init__(self, data, *, shape_type='rectangle', edge_width=1,
                 edge_color='black', face_color='white', opacity=1, z_index=0):

        self.shapes = []
        self._vertices = np.empty((0, 2))
        self._index = np.empty((0), dtype=int)
        self._z_index = np.empty((0), dtype=int)
        self._z_order = np.empty((0), dtype=int)

        self._mesh_vertices = np.empty((0, 2))
        self._mesh_vertices_centers = np.empty((0, 2))
        self._mesh_vertices_offsets = np.empty((0, 2))
        self._mesh_vertices_index = np.empty((0, 2), dtype=int)
        self._mesh_triangles = np.empty((0, 3), dtype=np.uint32)
        self._mesh_triangles_index = np.empty((0, 2), dtype=int)
        self._mesh_triangles_colors = np.empty((0, 4))
        self._mesh_triangles_z_order = np.empty((0), dtype=int)

        if issubclass(type(data), Shape):
            # If a single shape has been passed
            self.add(data)
        elif len(data) > 0:
            if issubclass(type(data[0]), Shape):
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
                    if type(shape_type) in (np.ndarray, list):
                        st = shape_type[i]
                    else:
                        st = shape_type
                    if type(edge_width) in (np.ndarray, list):
                        ew = edge_width[i]
                    else:
                        ew = edge_width
                    if type(edge_color) in (np.ndarray, list):
                        if np.isscalar(edge_color[i]):
                            ec = edge_color
                        else:
                            ec = edge_color[i]
                    else:
                        ec = edge_color
                    if type(face_color) in (np.ndarray, list):
                        if np.isscalar(face_color[i]):
                            fc = face_color
                        else:
                            fc = face_color[i]
                    else:
                        fc = face_color
                    if type(z_index) in (np.ndarray, list):
                        z = z_index[i]
                    else:
                        z = z_index
                    if type(opacity) in (np.ndarray, list):
                        o = opacity[i]
                    else:
                        o = opacity
                    self.add(d, shape_type=st, edge_width=ew, edge_color=ec,
                             face_color=fc, opacity=o, z_index=z)

    def add(self, data, shape_type='rectangle', edge_width=1,
            edge_color='black', face_color='white', opacity=1, z_index=0,
            shape_index=None):
        """Adds a single Shape object

        Parameters
        ----------
        data : np.ndarray | Shape
            Nx2 array of vertices or instance of the Shape class. If a Shape is
            passed the other parameters are ignored.
        shape_type : string
            String of shape shape_type, must be one of "{'line', 'rectangle',
            'ellipse', 'path', 'polygon'}".
        edge_width : float
            thickness of lines and edges.
        edge_color : str | tuple
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements.
        face_color : str | tuple
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements.
        opacity : float
            Opacity of the shape, must be between 0 and 1.
        z_index : int
            Specifier of z order priority. Shapes with higher z order are
            displayed ontop of others.
        shape : Shape
            An instance of the Shape class
        shape_index : None | int
            If int then edits the shape date at current index. To be used in
            conjunction with `remove` when renumber is `False`. If None, then
            appends a new shape to end of shapes list
        """
        if issubclass(type(data), Shape):
            shape = data
        else:
            if shape_type in self._shape_types.keys():
                shape_cls = self._shape_types[shape_type]
                shape = shape_cls(data, edge_width=edge_width,
                                  edge_color=edge_color, face_color=face_color,
                                  opacity=opacity, z_index=z_index)
            else:
                raise ValueError("""shape_type not recognized. Must be one of
                                 "{'line', 'rectangle', 'ellipse', 'path',
                                 'polygon'}".""")

        if shape_index is None:
            shape_index = len(self.shapes)
            self.shapes.append(shape)
            self._z_index = np.append(self._z_index, shape.z_index)
        else:
            self.shapes[shape_index] = shape
            self._z_index[shape_index] = shape.z_index

        self._vertices = np.append(self._vertices, shape.data, axis=0)
        index = np.repeat(shape_index, len(shape.data))
        self._index = np.append(self._index, index, axis=0)

        # Add edges to mesh
        m = len(self._mesh_vertices)
        vertices = shape._edge_vertices + shape.edge_width*shape._edge_offsets
        self._mesh_vertices = np.append(self._mesh_vertices, vertices, axis=0)
        vertices = shape._edge_vertices
        self._mesh_vertices_centers = np.append(self._mesh_vertices_centers,
                                                vertices, axis=0)
        vertices = shape._edge_offsets
        self._mesh_vertices_offsets = np.append(self._mesh_vertices_offsets,
                                                vertices, axis=0)
        index = np.repeat([[shape_index, 1]], len(vertices), axis=0)
        self._mesh_vertices_index = np.append(self._mesh_vertices_index, index,
                                              axis=0)

        triangles = shape._edge_triangles + m
        self._mesh_triangles = np.append(self._mesh_triangles, triangles,
                                         axis=0)
        index = np.repeat([[shape_index, 1]], len(triangles), axis=0)
        self._mesh_triangles_index = np.append(self._mesh_triangles_index,
                                               index, axis=0)
        color = shape.edge_color.rgba
        color[3] = color[3]*shape.opacity
        color_array = np.repeat([color], len(triangles), axis=0)
        self._mesh_triangles_colors = np.append(self._mesh_triangles_colors,
                                                color_array, axis=0)

        # Add faces to mesh
        m = len(self._mesh_vertices)
        vertices = shape._face_vertices
        self._mesh_vertices = np.append(self._mesh_vertices, vertices, axis=0)
        vertices = shape._face_vertices
        self._mesh_vertices_centers = np.append(self._mesh_vertices_centers,
                                                vertices, axis=0)
        vertices = np.zeros(shape._face_vertices.shape)
        self._mesh_vertices_offsets = np.append(self._mesh_vertices_offsets,
                                                vertices, axis=0)
        index = np.repeat([[shape_index, 0]], len(vertices), axis=0)
        self._mesh_vertices_index = np.append(self._mesh_vertices_index, index,
                                              axis=0)

        triangles = shape._face_triangles + m
        self._mesh_triangles = np.append(self._mesh_triangles, triangles,
                                         axis=0)
        index = np.repeat([[shape_index, 0]], len(triangles), axis=0)
        self._mesh_triangles_index = np.append(self._mesh_triangles_index,
                                               index, axis=0)
        color = shape.face_color.rgba
        color[3] = color[3]*shape.opacity
        color_array = np.repeat([color], len(triangles), axis=0)
        self._mesh_triangles_colors = np.append(self._mesh_triangles_colors,
                                                color_array, axis=0)

        # Set z_order
        self._update_z_order()

    def remove_all(self):
        """Removes all shapes
        """
        self.shapes = []
        self._vertices = np.empty((0, 2))
        self._index = np.empty((0), dtype=int)
        self._z_index = np.empty((0), dtype=int)
        self._z_order = np.empty((0), dtype=int)

        self._mesh_vertices = np.empty((0, 2))
        self._mesh_vertices_centers = np.empty((0, 2))
        self._mesh_vertices_offsets = np.empty((0, 2))
        self._mesh_vertices_index = np.empty((0, 2), dtype=int)
        self._mesh_triangles = np.empty((0, 3), dtype=np.uint32)
        self._mesh_triangles_index = np.empty((0, 2), dtype=int)
        self._mesh_triangles_colors = np.empty((0, 4))
        self._mesh_triangles_z_order = np.empty((0), dtype=int)

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
        self._mesh_triangles_index = self._mesh_triangles_index[indices]

        # Remove vertices
        indices = self._mesh_vertices_index[:, 0] != index
        self._mesh_vertices = self._mesh_vertices[indices]
        self._mesh_vertices_centers = self._mesh_vertices_centers[indices]
        self._mesh_vertices_offsets = self._mesh_vertices_offsets[indices]
        self._mesh_vertices_index = self._mesh_vertices_index[indices]
        indices = np.where(np.invert(indices))[0]
        num_indices = len(indices)
        if num_indices > 0:
            indices = self._mesh_triangles > indices[0]
            self._mesh_triangles[indices] = (self._mesh_triangles[indices] -
                                             num_indices)

        if renumber:
            del self.shapes[index]
            indices = self._index > index
            self._index[indices] = self._index[indices] - 1
            self._z_index = np.delete(self._z_index, index)
            indices = self._mesh_triangles_index[:, 0] > index
            self._mesh_triangles_index[indices, 0] = (
                self._mesh_triangles_index[indices, 0] - 1)
            indices = self._mesh_vertices_index[:, 0] > index
            self._mesh_vertices_index[indices, 0] = (
                self._mesh_vertices_index[indices, 0] - 1)
            self._update_z_order()

    def _update_mesh_vertices(self, index, edge=False, face=False):
        """Updates the mesh vertex data and vertex data for a single shape
        located at index.

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        edge : bool
            Bool to indicate whether to update mesh vertices corresponding to
            edges
        face : bool
            Bool to indicate whether to update mesh vertices corresponding to
            faces and to update the underlying shape vertices
        """
        shape = self.shapes[index]
        if edge:
            indices = np.all(self._mesh_vertices_index == [index, 1], axis=1)
            self._mesh_vertices[indices] = (shape._edge_vertices +
                                            shape.edge_width *
                                            shape._edge_offsets)
            self._mesh_vertices_centers[indices] = shape._edge_vertices
            self._mesh_vertices_offsets[indices] = shape._edge_offsets

        if face:
            indices = np.all(self._mesh_vertices_index == [index, 0], axis=1)
            self._mesh_vertices[indices] = shape._face_vertices
            self._mesh_vertices_centers[indices] = shape._face_vertices
            indices = self._index == index
            self._vertices[indices] = shape.data

    def _update_z_order(self):
        """Updates the z order of the triangles given the z_index list
        """
        self._z_order = np.argsort(self._z_index)[::-1]
        if len(self._z_order) == 0:
            self._mesh_triangles_z_order = np.empty((0), dtype=int)
        else:
            _, idx, counts = np.unique(self._mesh_triangles_index[:, 0],
                                       return_index=True, return_counts=True)
            triangles_z_order = ([np.arange(idx[z], idx[z]+counts[z]) for z in
                                 self._z_order])
            self._mesh_triangles_z_order = np.concatenate(triangles_z_order)

    def edit(self, index, data, new_type=None):
        """Updates the z order of a single shape located at index. If
        `new_type` is not None then converts the shape type to the new type

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        data : np.ndarray
            Nx2 array of vertices.
        new_type: None | str | Shape
            If string , must be one of "{'line', 'rectangle', 'ellipse',
            'path', 'polygon'}".
        """
        if new_type is not None:
            cur_shape = self.shapes[index]
            if type(new_type) == str:
                if new_type in self._shape_types.keys():
                    shape_cls = self._shape_types[new_type]
                else:
                    raise ValueError("""shape_type not recognized. Must be one of
                                 "{'line', 'rectangle', 'ellipse', 'path',
                                 'polygon'}".""")
            else:
                shape_cls = new_type
            shape = shape_cls(data, edge_width=cur_shape.edge_width,
                              edge_color=cur_shape.edge_color,
                              face_color=cur_shape.face_color,
                              opacity=cur_shape.opacity,
                              z_index=cur_shape.z_index)
        else:
            shape = self.shapes[index]
            shape.data = data

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
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements.
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
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements.
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
            Specifier of z order priority. Shapes with higher z order are
            displayed ontop of others.
        """
        self.shapes[index].z_index = z_index
        self._z_index[index] = z_index
        self._update_z_order()

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
        shape = self.shapes[index]
        self.remove(index, renumber=False)
        self.add(shape, shape_index=index)

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
        shape = self.shapes[index]
        self.remove(index, renumber=False)
        self.add(shape, shape_index=index)

    def to_list(self, shape_type=None):
        """Returns the vertex data assoicated with the shapes as a list
        where each element of the list corresponds to one shape. Passing a
        `shape_type` argument leads to only that particular `shape_type`
        being returned.

        Parameters
        ----------
        shape_type : str
            String of shape shape_type, must be one of "{'line', 'rectangle',
            'ellipse', 'path', 'polygon'}".

        Returns
        ----------
        data : list
            List of shape data where each element of the list is an
            `np.ndarray` corresponding to one shape
        """
        if shape_type is None:
            data = [s.data for s in self.shapes]
        elif shape_type not in self._shape_types.keys():
            raise ValueError("""shape_type not recognized, must be one of
                         "{'line', 'rectangle', 'ellipse', 'path',
                         'polygon'}"
                         """)
        else:
            cls = self._shape_types[shape_type]
            data = [s.data for s in self.shapes if type(s) == cls]
        return data
