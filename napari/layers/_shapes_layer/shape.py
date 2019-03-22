import numpy as np
from vispy.color import Color

from .shape_util import (triangulate_edge, triangulate_ellipse,
                         triangulate_face, center_radii_to_corners,
                         find_corners, rectangle_to_box, create_box)


class Shape():
    """Class for a single shape
    Parameters
    ----------
    data : np.ndarray
        Nx2 array of vertices specifying the shape.
    shape_type : string
        String of shape shape_type, must be one of "{'line', 'rectangle',
        'ellipse', 'path', 'polygon'}".
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

    Attributes
    ----------
    data : np.ndarray
        Nx2 array of vertices specifying the shape.
    shape_type : string
        String of shape shape_type, must be one of "{'line', 'rectangle',
        'ellipse', 'path', 'polygon'}".
    edge_width : float
        thickness of lines and edges.
    edge_color : ColorArray
        Color of the shape edge
    face_color : ColorArray
        Color of the shape face
    opacity : float
        Opacity of the shape, must be between 0 and 1.
    z_index : int
        Specifier of z order priority. Shapes with higher z order are displayed
        ontop of others.
    _box : np.ndarray
        9x2 array of vertices of the interaction box. The first 8 points are
        the corners and midpoints of the box in clockwise order starting in the
        upper-left corner. The last point is the center of the box
    _face_vertices : np.ndarray
        Qx2 array of vertices of all triangles for the shape face
    _face_triangles : np.ndarray
        Px3 array of vertex indices that form the triangles for the shape face
    _edge_vertices : np.ndarray
        Rx2 array of centers of vertices of triangles for the shape edge.
        These values should be added to the scaled `_edge_offsets` to get the
        actual vertex positions. The scaling corresponds to the width of the
        edge
    _edge_offsets : np.ndarray
        Sx2 array of offsets of vertices of triangles for the shape edge. For
        These values should be scaled and added to the `_edge_vertices` to get
        the actual vertex positions. The scaling corresponds to the width of
        the edge
    _edge_triangles : np.ndarray
        Tx3 array of vertex indices that form the triangles for the shape edge
    _shape_types : list
        List of the supported shape types
    """
    _shape_types = ['line', 'rectangle', 'ellipse', 'path', 'polygon']

    def __init__(self, data, shape_type='rectangle', edge_width=1,
                 edge_color='black', face_color='white', opacity=1, z_index=0):

        self._face_vertices = np.empty((0, 2))
        self._face_triangles = np.empty((0, 3), dtype=np.uint32)
        self._edge_vertices = np.empty((0, 2))
        self._edge_offsets = np.empty((0, 2))
        self._edge_triangles = np.empty((0, 3), dtype=np.uint32)
        self._box = np.empty((9, 2))

        self.shape_type = shape_type
        self.data = np.array(data)
        self.edge_width = edge_width
        self.edge_color = edge_color
        self.face_color = face_color
        self.opacity = opacity
        self.z_index = z_index

    @property
    def shape_type(self):
        """string: shape_type, must be one of "{'line', 'rectangle', 'ellipse',
            'path', 'polygon'}".
        """
        return self._shape_type

    @shape_type.setter
    def shape_type(self, shape_type):
        if shape_type not in self._shape_types:
            raise ValueError("""shape_type not recognized, must be one of
                             "{'line', 'rectangle', 'ellipse', 'path',
                             'polygon'}"
                             """)
        self._shape_type = shape_type

    @property
    def data(self):
        """np.ndarray: Nx2 array of vertices.
        """
        return self._data

    @data.setter
    def data(self, data):
        if self.shape_type == 'line':
            if len(data) != 2:
                raise ValueError("""Data shape does not match a line. Line
                                 expects two end vertices""")
            else:
                # For line connect two points
                self._set_meshes(data, face=False, closed=False)
                self._box = create_box(data)
        elif self.shape_type == 'rectangle':
            if len(data) == 2:
                data = find_corners(data)
            if len(data) != 4:
                raise ValueError("""Data shape does not match a rectangle.
                                 Rectangle expects four corner vertices""")
            else:
                # Add four boundary lines and then two triangles for each
                self._set_meshes(data, face=False)
                self._face_vertices = data
                self._face_triangles = np.array([[0, 1, 2], [0, 2, 3]])
                self._box = rectangle_to_box(data)
        elif self.shape_type == 'ellipse':
            if len(data) == 2:
                data = center_radii_to_corners(data[0], data[1])
            if len(data) != 4:
                raise ValueError("""Data shape does not match an ellipse.
                                 Ellipse expects four corner vertices""")
            else:
                # Build boundary vertices with num_segments
                vertices, trinalges = triangulate_ellipse(data)
                self._set_meshes(vertices[1:-1], face=False)
                self._face_vertices = vertices
                self._face_triangles = trinalges
                self._box = rectangle_to_box(data)
        elif self.shape_type == 'path':
            if len(data) < 2:
                raise ValueError("""Data shape does not match a path. Path
                                 expects at least two vertices""")
            else:
                # For path connect every all data
                self._set_meshes(data, face=False, closed=False)
                self._box = create_box(data)
        elif self.shape_type == 'polygon':
            if len(data) < 2:
                raise ValueError("""Data shape does not match a polygon.
                                 Polygon expects at least two vertices""")
            else:
                self._set_meshes(data)
                self._box = create_box(data)
        else:
            raise ValueError("""Shape shape_type not recognized, must be one of
                             "{'line', 'rectangle', 'ellipse', 'path',
                             'polygon'}"
                             """)
        create_box(data)
        self._data = data

    @property
    def edge_width(self):
        """float: thickness of lines and edges.
        """
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width):
        self._edge_width = edge_width

    @property
    def edge_color(self):
        """Color, ColorArray: color of edges
        """
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color):
        self._edge_color = Color(edge_color)

    @property
    def face_color(self):
        """Color, ColorArray: color of faces
        """
        return self._face_color

    @face_color.setter
    def face_color(self, face_color):
        self._face_color = Color(face_color)

    @property
    def opacity(self):
        """float: opacity of shape
        """
        return self._opacity

    @opacity.setter
    def opacity(self, opacity):
        self._opacity = opacity

    @property
    def z_index(self):
        """int: z order priority of shape. Shapes with higher z order displayed
        ontop of others.
        """
        return self._z_index

    @z_index.setter
    def z_index(self, z_index):
        self._z_index = z_index

    def _set_meshes(self, data, closed=True, face=True, edge=True):
        """Sets the face and edge meshes from a set of points.

        Parameters
        ----------
        data : np.ndarray
            Nx2 array specifying the shape to be triangulated
        closed : bool
            Bool which determines if the edge is closed or not
        face : bool
            Bool which determines if the face need to be traingulated
        edge : bool
            Bool which determines if the edge need to be traingulated
        """
        if edge:
            centers, offsets, triangles = triangulate_edge(data, closed=closed)
            self._edge_vertices = centers
            self._edge_offsets = offsets
            self._edge_triangles = triangles
        if face:
            if len(data) > 2:
                vertices, triangles = triangulate_face(data)
                if len(triangles) > 0:
                    self._face_vertices = vertices
                    self._face_triangles = triangles

    def transform(self, transform):
        """Perfroms a linear transform on the shape

        Parameters
        ----------
        transform : np.ndarray
            2x2 array specifying linear transform.
        """
        self._box = np.matmul(self._box, transform.T)
        self._data = np.matmul(self._data, transform.T)
        self._face_vertices = np.matmul(self._face_vertices, transform.T)

        if self.shape_type == 'path' or self.shape_type == 'line':
            closed = False
        else:
            closed = True

        if self.shape_type == 'ellipse':
            points = self._face_vertices[1:-1]
        else:
            points = self._data

        centers, offsets, triangles = triangulate_edge(points, closed=closed)
        self._edge_vertices = centers
        self._edge_offsets = offsets
        self._edge_triangles = triangles

    def shift(self, shift):
        """Perfroms a 2D shift on the shape

        Parameters
        ----------
        shift : np.ndarray
            length 2 array specifying shift of shapes.
        """
        shift = np.array(shift)

        self._face_vertices = self._face_vertices + shift
        self._edge_vertices = self._edge_vertices + shift
        self._box = self._box + shift
        self._data = self._data + shift

    def scale(self, scale, center=None):
        """Perfroms a scaling on the shape

        Parameters
        ----------
        scale : float, list
            scalar or list specifying rescaling of shape.
        center : list
            length 2 list specifying coordinate of center of scaling.
        """
        if isinstance(scale, (list, np.ndarray)):
            transform = np.array([[scale[0], 0], [0, scale[1]]])
        else:
            transform = np.array([[scale, 0], [0, scale]])
        if center is None:
            self.transform(transform)
        else:
            self.shift(-center)
            self.transform(transform)
            self.shift(center)

    def rotate(self, angle, center=None):
        """Perfroms a rotation on the shape

        Parameters
        ----------
        angle : float
            angle specifying rotation of shape in degrees.
        center : list
            length 2 list specifying coordinate of center of rotation.
        """
        theta = np.radians(angle)
        transform = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
        if center is None:
            self.transform(transform)
        else:
            self.shift(-center)
            self.transform(transform)
            self.shift(center)

    def flip(self, axis, center=None):
        """Perfroms an vertical flip on the shape

        Parameters
        ----------
        axis : int
            integer specifying axis of flip. `0` flips horizontal, `1` flips
            vertical.
        center : list
            length 2 list specifying coordinate of center of flip axes.
        """
        if axis == 0:
            transform = np.array([[1, 0], [0, -1]])
        elif axis == 1:
            transform = np.array([[-1, 0], [0, 1]])
        else:
            raise ValueError("""Axis not recognized, must be one of "{0, 1}"
                             """)
        if center is None:
            self.transform(transform)
        else:
            self.shift(-center)
            self.transform(transform)
            self.shift(-center)
