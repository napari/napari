import numpy as np
from xml.etree.ElementTree import Element
from .shape import Shape
from ..shape_utils import create_box


class Line(Shape):
    """Class for a single line segment

    Parameters
    ----------
    data : (2, D) array
        Line vertices.
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
        self._filled = False
        self.data = data
        self.name = 'line'

    @property
    def data(self):
        """(2, D) array: line vertices.
        """
        return self._data

    @data.setter
    def data(self, data):
        data = np.array(data).astype(float)

        if len(self.dims_order) != data.shape[1]:
            self._dims_order = list(range(data.shape[1]))

        if len(data) != 2:
            raise ValueError(
                f"""Data shape does not match a line. A
                             line expects two end vertices,
                             {len(data)} provided."""
            )

        self._data = data
        self._update_displayed_data()

    def _update_displayed_data(self):
        """Update the data that is to be displayed."""
        # For path connect every all data
        self._set_meshes(self.data_displayed, face=False, closed=False)
        self._box = create_box(self.data_displayed)

        data_not_displayed = self.data[:, self.dims_not_displayed]
        self.slice_key = np.round(
            [
                np.min(data_not_displayed, axis=0),
                np.max(data_not_displayed, axis=0),
            ]
        ).astype('int')

    def to_xml(self):
        """Generates an xml element that defintes the shape according to the
        svg specification.

        Returns
        ----------
        element : xml.etree.ElementTree.Element
            xml element specifying the shape according to svg.
        """
        data = self.data[:, self.dims_displayed]
        x1 = str(data[0, 0])
        y1 = str(data[0, 1])
        x2 = str(data[1, 0])
        y2 = str(data[1, 1])

        element = Element('line', x1=y1, y1=x1, x2=y2, y2=x2, **self.svg_props)

        return element
