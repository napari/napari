import numpy as np

from napari.layers.shapes._shapes_models.shape import Shape
from napari.layers.shapes._shapes_utils import create_box
from napari.utils.translations import trans


class PolygonBase(Shape):
    """Class for a polygon or path.

    Parameters
    ----------
    data : np.ndarray
        NxD array of vertices specifying the path.
    edge_width : float
        thickness of lines and edges.
    z_index : int
        Specifier of z order priority. Shapes with higher z order are displayed
        ontop of others.
    dims_order : (D,) list
        Order that the dimensions are to be rendered in.
    closed : bool
        Bool if shape edge is a closed path or not.
    filled : bool
        Flag if array is filled or not.
    name : str
        Name of the shape.
    """

    def __init__(
        self,
        data,
        *,
        edge_width=1,
        z_index=0,
        dims_order=None,
        ndisplay=2,
        filled=True,
        closed=True,
        name='polygon',
    ):

        super().__init__(
            edge_width=edge_width,
            z_index=z_index,
            dims_order=dims_order,
            ndisplay=ndisplay,
        )
        self._filled = filled
        self._closed = closed
        self.data = data
        self.name = name

    @property
    def data(self):
        """np.ndarray: NxD array of vertices."""
        return self._data

    @data.setter
    def data(self, data):
        data = np.array(data).astype(float)

        if len(self.dims_order) != data.shape[1]:
            self._dims_order = list(range(data.shape[1]))

        if len(data) < 2:
            raise ValueError(
                trans._(
                    "Shape needs at least two vertices, {number} provided.",
                    deferred=True,
                    number=len(data),
                )
            )

        self._data = data
        self._update_displayed_data()

    def _update_displayed_data(self):
        """Update the data that is to be displayed."""
        # For path connect every all data
        self._set_meshes(
            self.data_displayed, face=self._filled, closed=self._closed
        )
        self._box = create_box(self.data_displayed)

        data_not_displayed = self.data[:, self.dims_not_displayed]
        self.slice_key = np.round(
            [
                np.min(data_not_displayed, axis=0),
                np.max(data_not_displayed, axis=0),
            ]
        ).astype('int')
