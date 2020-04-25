import numpy as np
from .shape import Shape
from .._shapes_utils import find_corners, rectangle_to_box


class Rectangle(Shape):
    """Class for a single rectangle

    Parameters
    ----------
    data : (4, D) or (2, 2) array
        Either a (2, 2) array specifying the two corners of an axis aligned
        rectangle, or a (4, D) array specifying the four corners of a bounding
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
        self.data = data
        self.name = 'rectangle'

    @property
    def data(self):
        """(4, D) array: rectangle vertices.
        """
        return self._data

    @data.setter
    def data(self, data):
        data = np.array(data).astype(float)

        if len(self.dims_order) != data.shape[1]:
            self._dims_order = list(range(data.shape[1]))

        if len(data) == 2 and data.shape[1] == 2:
            data = find_corners(data)

        if len(data) != 4:
            print(data)
            raise ValueError(
                f"""Data shape does not match a rectangle.
                             Rectangle expects four corner vertices,
                             {len(data)} provided."""
            )

        self._data = data
        self._update_displayed_data()

    def _update_displayed_data(self):
        """Update the data that is to be displayed."""
        # Add four boundary lines and then two triangles for each
        self._set_meshes(self.data_displayed, face=False)
        self._face_vertices = self.data_displayed
        self._face_triangles = np.array([[0, 1, 2], [0, 2, 3]])
        self._box = rectangle_to_box(self.data_displayed)

        data_not_displayed = self.data[:, self.dims_not_displayed]
        self.slice_key = np.round(
            [
                np.min(data_not_displayed, axis=0),
                np.max(data_not_displayed, axis=0),
            ]
        ).astype('int')
