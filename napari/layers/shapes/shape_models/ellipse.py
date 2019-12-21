import numpy as np
from xml.etree.ElementTree import Element
from .shape import Shape
from ..shape_utils import (
    triangulate_edge,
    triangulate_ellipse,
    center_radii_to_corners,
    rectangle_to_box,
)


class Ellipse(Shape):
    """Class for a single ellipse

    Parameters
    ----------
    data : (4, D) array or (2, 2) array.
        Either a (2, 2) array specifying the center and radii of an axis
        aligned ellipse, or a (4, D) array specifying the four corners of a
        boudning box that contains the ellipse. These need not be axis aligned.
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
    dims_order : (D,) list
        Order that the dimensions are to be rendered in.
    """

    def __init__(
        self,
        data,
        *,
        edge_width=1,
        edge_color='black',
        face_color='white',
        opacity=1,
        z_index=0,
        dims_order=None,
        ndisplay=2,
    ):

        super().__init__(
            edge_width=edge_width,
            edge_color=edge_color,
            face_color=face_color,
            opacity=opacity,
            z_index=z_index,
            dims_order=dims_order,
            ndisplay=ndisplay,
        )

        self._closed = True
        self._use_face_vertices = True
        self.data = data
        self.name = 'ellipse'

    @property
    def data(self):
        """(4, D) array: ellipse vertices.
        """
        return self._data

    @data.setter
    def data(self, data):
        data = np.array(data).astype(float)

        if len(self.dims_order) != data.shape[1]:
            self._dims_order = list(range(data.shape[1]))

        if len(data) == 2 and data.shape[1] == 2:
            data = center_radii_to_corners(data[0], data[1])

        if len(data) != 4:
            raise ValueError(
                f"""Data shape does not match a ellipse.
                             Ellipse expects four corner vertices,
                             {len(data)} provided."""
            )

        self._data = data
        self._update_displayed_data()

    def _update_displayed_data(self):
        """Update the data that is to be displayed."""
        # Build boundary vertices with num_segments
        vertices, triangles = triangulate_ellipse(self.data_displayed)
        self._set_meshes(vertices[1:-1], face=False)
        self._face_vertices = vertices
        self._face_triangles = triangles
        self._box = rectangle_to_box(self.data_displayed)

        data_not_displayed = self.data[:, self.dims_not_displayed]
        self.slice_key = np.round(
            [
                np.min(data_not_displayed, axis=0),
                np.max(data_not_displayed, axis=0),
            ]
        ).astype('int')

    def transform(self, transform):
        """Performs a linear transform on the shape

        Parameters
        ----------
        transform : np.ndarray
            2x2 array specifying linear transform.
        """
        self._box = self._box @ transform.T
        self._data[:, self.dims_displayed] = (
            self._data[:, self.dims_displayed] @ transform.T
        )
        self._face_vertices = self._face_vertices @ transform.T

        points = self._face_vertices[1:-1]

        centers, offsets, triangles = triangulate_edge(
            points, closed=self._closed
        )
        self._edge_vertices = centers
        self._edge_offsets = offsets
        self._edge_triangles = triangles

    def to_xml(self):
        """Generates an xml element that defintes the shape according to the
        svg specification.

        Returns
        ----------
        element : xml.etree.ElementTree.Element
            xml element specifying the shape according to svg.
        """
        props = self.svg_props
        data = self.data[:, self.dims_displayed[::-1]]

        offset = data[1] - data[0]
        angle = -np.arctan2(offset[0], -offset[1])
        if not angle == 0:
            # if shape has been rotated, shift to origin
            cen = data.mean(axis=0)
            coords = data - cen

            # rotate back to axis aligned
            c, s = np.cos(angle), np.sin(-angle)
            rotation = np.array([[c, s], [-s, c]])
            coords = coords @ rotation.T

            # shift back to center
            coords = coords + cen

            # define rotation around center
            transform = f'rotate({np.degrees(-angle)} {cen[0]} {cen[1]})'
            props['transform'] = transform
        else:
            coords = data

        cx = str(cen[0])
        cy = str(cen[1])
        size = abs(coords[2] - coords[0])
        rx = str(size[0] / 2)
        ry = str(size[1] / 2)

        element = Element('ellipse', cx=cx, cy=cy, rx=rx, ry=ry, **props)
        return element
