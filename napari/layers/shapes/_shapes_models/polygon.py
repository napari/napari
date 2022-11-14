from napari.layers.shapes._shapes_models._polgyon_base import PolygonBase


class Polygon(PolygonBase):
    """Class for a single polygon

    Parameters
    ----------
    data : np.ndarray
        NxD array of vertices specifying the shape.
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
        interpolation_order=1,
    ):

        super().__init__(
            data=data,
            edge_width=edge_width,
            z_index=z_index,
            dims_order=dims_order,
            ndisplay=ndisplay,
            closed=True,
            filled=True,
            name='polygon',
        )
