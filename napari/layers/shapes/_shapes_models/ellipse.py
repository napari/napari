import numpy as np

from napari.layers.shapes._shapes_models.shape import Shape
from napari.layers.shapes._shapes_utils import (
    center_radii_to_corners,
    rectangle_to_box,
    triangulate_edge,
    triangulate_ellipse,
)
from napari.utils.translations import trans


class Ellipse(Shape):
    """Class for a single ellipse

    Parameters
    ----------
    data : (4, D) array or (2, 2) array.
        Either a (2, 2) array specifying the center and radii of an axis
        aligned ellipse, or a (4, D) array specifying the four corners of a
        bounding box that contains the ellipse. These need not be axis aligned.
    edge_width : float
        thickness of lines and edges.
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
        opacity=1,
        z_index=0,
        dims_order=None,
        ndisplay=2,
    ) -> None:

        super().__init__(
            edge_width=edge_width,
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
        """(4, D) array: ellipse vertices."""
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
                trans._(
                    "Data shape does not match a ellipse. Ellipse expects four corner vertices, {number} provided.",
                    deferred=True,
                    number=len(data),
                )
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
