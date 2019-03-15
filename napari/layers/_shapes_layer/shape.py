import numpy as np
from vispy.geometry import PolygonData
from vispy.color import Color
from copy import copy

from .shape_utils import triangulate_path, create_box, generate_ellipse, expand_ellipse, expand_rectangle, expand_box

class Shape():
    """Class for a single shape
    Parameters
    ----------
    data : np.ndarray
        Nx2 array of vertices.
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
    z_order : int
        Specifier of z order priority. Shapes with higher z order are displayed
        ontop of others.
    """
    _ellipse_segments = 100
    _shape_types = ['line', 'rectangle', 'ellipse', 'path', 'polygon']

    def __init__(self, data, shape_type='rectangle', edge_width=1, edge_color='black',
                 face_color='white', z_order=0):

        self._face_vertices = np.empty((0, 2)) # Mx2 array of vertices of faces
        self._face_triangles = np.empty((0, 3), dtype=np.uint32) # Px3 array of vertex indices that form triangles for faces
        self._edge_vertices = np.empty((0, 2)) # Mx2 array of vertices of edges
        self._edge_offsets = np.empty((0, 2)) # Mx2 array of vertex offsets of edges
        self._edge_triangles = np.empty((0, 3), dtype=np.uint32) # Px3 array of vertex indices that form triangles for edges
        self._box = np.empty((9, 2)) # 8 vertex bounding box and center point

        self.shape_type = shape_type
        self.data = data
        self.edge_width = edge_width
        self.edge_color = edge_color
        self.face_color = face_color
        self.z_order = z_order

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
                self._set_meshes(data, fill=False, closed=False)
                self._box = create_box(data)
        elif self.shape_type == 'rectangle':
            if len(data) == 2:
                data = expand_rectangle(data)
            if len(data) != 4:
                raise ValueError("""Data shape does not match a rectangle.
                                 Rectangle expects four corner vertices""")
            else:
                # Add four boundary lines and then two triangles for each
                fill_triangles = np.array([[0, 1, 2], [0, 2, 3]])
                self._set_meshes(data, fill_vertices=data,
                                  fill_triangles=fill_triangles)
                self._box = expand_box(data)
        elif self.shape_type == 'ellipse':
            if len(data) == 2:
                data = expand_ellipse(data)
            if len(data) != 4:
                raise ValueError("""Data shape does not match an ellipse.
                                 Ellipse expects four corner vertices""")
            else:
                # Build boundary vertices with num_segments
                points = generate_ellipse(data, self._ellipse_segments)
                fill_triangles = np.array([[0, i+1, i+2] for i in range(self._ellipse_segments)])
                fill_triangles[-1, 2] = 1
                self._set_meshes(points[1:-1], fill_vertices=points,
                                  fill_triangles=fill_triangles)
                self._box = expand_box(data)
        elif self.shape_type == 'path':
            if len(data) < 2:
                raise ValueError("""Data shape does not match a path. Path
                                 expects at least two vertices""")
            else:
                # For path connect every all data
                self._set_meshes(data, fill=False, closed=False)
                self._box = create_box(data)
        elif self.shape_type == 'polygon':
            if len(data) < 2:
                raise ValueError("""Data shape does not match a polygon. Polygon
                                 expects at least two vertices""")
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
    def z_order(self):
        """int: z order priority of shape. Shapes with higher z order displayed
        ontop of others.
        """
        return self._z_order

    @z_order.setter
    def z_order(self, z_order):
        self._z_order = z_order

    def _set_meshes(self, points, closed=True, fill=True, edge=True,
                     fill_vertices=None, fill_triangles=None):
        if edge:
            centers, offsets, triangles = triangulate_path(points, closed=closed)
            self._edge_vertices = centers
            self._edge_offsets = offsets
            self._edge_triangles = triangles
        if fill:
            if fill_vertices is None or fill_triangles is None:
                if len(points) > 2:
                    vertices, faces = PolygonData(vertices=points).triangulate()
                    if len(faces) > 0:
                        self._face_vertices = vertices
                        self._face_triangles = faces.astype(np.uint32)
            else:
                self._face_vertices = fill_vertices
                self._face_triangles = fill_triangles

    def transform(self, transform):
        """Perfroms a linear transform on the shape
        Parameters
        ----------
        transform : np.ndarray
            2x2 array specifying linear transform.
        """
        A = transform.T
        self._face_vertices = np.matmul(self._face_vertices, A)
        self._edge_vertices = np.matmul(self._edge_vertices, A)

        norm_offsets = np.linalg.norm(self._edge_offsets, axis=1, keepdims=True)
        offsets = np.matmul(self._edge_offsets, A)
        transformed_norm_offsets = np.linalg.norm(offsets, axis=1, keepdims=True)
        norm_offsets[transformed_norm_offsets==0] = 1
        transformed_norm_offsets[transformed_norm_offsets==0] = 1
        self._edge_offsets = offsets/transformed_norm_offsets*norm_offsets

        self._box = np.matmul(self._box, A)
        self._data = np.matmul(self._data, A)

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
        if center is None:
            transform = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
            self.transform(transform)
        else:
            self.shift(-center)
            transform = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
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
