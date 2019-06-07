from abc import ABC, abstractmethod
import numpy as np
from vispy.color import Color
from ..shape_util import triangulate_edge, triangulate_face, is_collinear


class Shape(ABC):
    """Base class for a single shape

    Parameters
    ----------
    data : np.ndarray
        Nx2 array of vertices specifying the shape.
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
    edge_width : float
        thickness of lines and edges.
    edge_color : ColorArray
        Color of the shape edge
    face_color : ColorArray
        Color of the shape face
    opacity : float
        Opacity of the shape, must be between 0 and 1.
    name : str
        Name of shape type.
    z_index : int
        Specifier of z order priority. Shapes with higher z order are displayed
        ontop of others.

    Extended Summary
    ----------
    _edge_color_name : str
        Name of edge color or six digit hex code representing edge color if not
        recongnized
    _face_color_name : str
        Name of edge color or six digit hex code representing face color if not
        recongnized
    _closed : bool
        Bool if shape edge is a closed path or not
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
    """

    def __init__(
        self,
        *,
        shape_type='rectangle',
        edge_width=1,
        edge_color='black',
        face_color='white',
        opacity=1,
        z_index=0,
    ):

        self._face_vertices = np.empty((0, 2))
        self._face_triangles = np.empty((0, 3), dtype=np.uint32)
        self._edge_vertices = np.empty((0, 2))
        self._edge_offsets = np.empty((0, 2))
        self._edge_triangles = np.empty((0, 3), dtype=np.uint32)
        self._box = np.empty((9, 2))
        self._edge_color_name = 'black'
        self._face_color_name = 'white'

        self._closed = False
        self.edge_width = edge_width
        self.edge_color = edge_color
        self.face_color = face_color
        self.opacity = opacity
        self.z_index = z_index
        self.name = ''

    @property
    @abstractmethod
    def data(self):
        # user writes own docstring
        raise NotImplementedError()

    @data.setter
    @abstractmethod
    def data(self, data):
        raise NotImplementedError()

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
        if type(edge_color) is str:
            self._edge_color_name = edge_color
        else:
            rgb = tuple([int(255 * x) for x in self._edge_color.rgba[:3]])
            self._edge_color_name = '#%02x%02x%02x' % rgb

    @property
    def face_color(self):
        """Color, ColorArray: color of faces
        """
        return self._face_color

    @face_color.setter
    def face_color(self, face_color):
        self._face_color = Color(face_color)
        if type(face_color) is str:
            self._face_color_name = face_color
        else:
            rgb = tuple([int(255 * x) for x in self._face_color.rgba[:3]])
            self._face_color_name = '#%02x%02x%02x' % rgb

    @property
    def opacity(self):
        """float: opacity of shape
        """
        return self._opacity

    @opacity.setter
    def opacity(self, opacity):
        self._opacity = opacity

    @property
    def svg_props(self):
        """dict: color and width properties in the svg specification
        """
        width = str(self.edge_width)
        face_color = (255 * self.face_color.rgba).astype(np.int)
        fill = f'rgb{tuple(face_color[:3])}'
        edge_color = (255 * self.edge_color.rgba).astype(np.int)
        stroke = f'rgb{tuple(edge_color[:3])}'
        opacity = str(self.opacity)

        # Currently not using fill or stroke opacity - only global opacity
        # as otherwise leads to unexpected behavior when reading svg into
        # other applications
        # fill_opacity = f'{self.opacity*self.face_color.rgba[3]}'
        # stroke_opacity = f'{self.opacity*self.edge_color.rgba[3]}'

        props = {
            'fill': fill,
            'stroke': stroke,
            'stroke-width': width,
            'opacity': opacity,
        }

        return props

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
            clean_data = np.array(
                [
                    p
                    for i, p in enumerate(data)
                    if i == 0 or not np.all(p == data[i - 1])
                ]
            )
            if not is_collinear(clean_data):
                vertices, triangles = triangulate_face(clean_data)
                if len(triangles) > 0:
                    self._face_vertices = vertices
                    self._face_triangles = triangles

    def transform(self, transform):
        """Performs a linear transform on the shape

        Parameters
        ----------
        transform : np.ndarray
            2x2 array specifying linear transform.
        """
        self._box = self._box @ transform.T
        self._data = self._data @ transform.T
        self._face_vertices = self._face_vertices @ transform.T

        points = self._data

        centers, offsets, triangles = triangulate_edge(
            points, closed=self._closed
        )
        self._edge_vertices = centers
        self._edge_offsets = offsets
        self._edge_triangles = triangles

    def shift(self, shift):
        """Performs a 2D shift on the shape

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
        """Performs a scaling on the shape

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
        """Performs a rotation on the shape

        Parameters
        ----------
        angle : float
            angle specifying rotation of shape in degrees. CCW is positive.
        center : list
            length 2 list specifying coordinate of fixed point of the rotation.
        """
        theta = np.radians(angle)
        transform = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )
        if center is None:
            self.transform(transform)
        else:
            self.shift(-center)
            self.transform(transform)
            self.shift(center)

    def flip(self, axis, center=None):
        """Performs a flip on the shape, either horizontal or vertical.

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
            raise ValueError(
                """Axis not recognized, must be one of "{0, 1}"
                             """
            )
        if center is None:
            self.transform(transform)
        else:
            self.shift(-center)
            self.transform(transform)
            self.shift(-center)

    @abstractmethod
    def to_mask(self, mask_shape=None):
        # user writes own docstring
        raise NotImplementedError()

    @abstractmethod
    def to_xml(self):
        # user writes own docstring
        raise NotImplementedError()
