import numpy as np
from xml.etree.ElementTree import Element
from .shape import Shape
from ..shape_util import find_corners, rectangle_to_box, poly_to_mask


class Rectangle(Shape):
    """Class for a single rectangle

    Parameters
    ----------
    data : (4, 2) or (2, 2) array
        Either a (2, 2) array specifying the two corners of an axis aligned
        rectangle, or a (4, 2) array specifying the four corners of a bounding
        box that contains the rectangle. These need not be axis aligned.
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
    ):

        super().__init__(
            edge_width=edge_width,
            edge_color=edge_color,
            face_color=face_color,
            opacity=opacity,
            z_index=z_index,
        )

        self._closed = True
        self.data = np.array(data)
        self.name = 'rectangle'

    @property
    def data(self):
        """(4, 2) array: rectangle vertices.
        """
        return self._data

    @data.setter
    def data(self, data):
        if len(data) == 2:
            data = find_corners(data)
        if len(data) != 4:
            raise ValueError(
                """Data shape does not match a rectangle.
                             Rectangle expects four corner vertices"""
            )
        else:
            # Add four boundary lines and then two triangles for each
            self._set_meshes(data, face=False)
            self._face_vertices = data
            self._face_triangles = np.array([[0, 1, 2], [0, 2, 3]])
            self._box = rectangle_to_box(data)

        self._data = data

    def to_mask(self, mask_shape=None):
        """Converts the shape vertices to a boolean mask with `True` for points
        lying inside the shape.

        Parameters
        ----------
        mask_shape : np.ndarray | tuple | None
            1x2 array of shape of mask to be generated. If non specified, takes
            the max of the vertiecs

        Returns
        ----------
        mask : np.ndarray
            Boolean array with `True` for points inside the shape
        """
        if mask_shape is None:
            mask_shape = self.data.max(axis=0).astype('int')

        mask = poly_to_mask(mask_shape, self.data)

        return mask

    def to_xml(self):
        """Generates an xml element that defintes the shape according to the
        svg specification.

        Returns
        ----------
        element : xml.etree.ElementTree.Element
            xml element specifying the shape according to svg.
        """
        props = self.svg_props
        data = self.data[:, ::-1]

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

        x = str(coords.min(axis=0)[0])
        y = str(coords.min(axis=0)[1])
        size = abs(coords[2] - coords[0])
        width = str(size[0])
        height = str(size[1])

        element = Element(
            'rect', x=x, y=y, width=width, height=height, **props
        )
        return element
