import numpy as np

from napari.layers.shapes._shapes_utils import create_box
from napari.utils.translations import trans

from .shape import Shape


class Line(Shape):
    """Class for a single line segment

    Parameters
    ----------
    data : (2, D) array
        Line vertices.
    edge_width : float
        thickness of lines and edges.
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
        z_index=0,
        dims_order=None,
        ndisplay=2,
    ):

        super().__init__(
            edge_width=edge_width,
            z_index=z_index,
            dims_order=dims_order,
            ndisplay=ndisplay,
        )
        self._filled = False
        self.data = data
        self.name = 'line'

    @property
    def data(self):
        """(2, D) array: line vertices."""
        return self._data

    @data.setter
    def data(self, data):
        data = np.array(data).astype(float)

        if len(self.dims_order) != data.shape[1]:
            self._dims_order = list(range(data.shape[1]))

        if len(data) != 2:
            raise ValueError(
                trans._(
                    "Data shape does not match a line. A line expects two end vertices, {number} provided.",
                    deferred=True,
                    number=len(data),
                )
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
