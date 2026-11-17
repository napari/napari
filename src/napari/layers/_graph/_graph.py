import numpy as np
from networkx import Graph as nxGraph

from napari.layers._graph._graph_utils import fix_graph_data
from napari.layers._graph._slice import _GraphSlicingState
from napari.layers.base import Layer


class _Graph(Layer):
    """Graph layer.

    Parameters
    ----------
    data : nx.Graph, optional
        The graph data to be displayed in the layer. If None, an empty graph is created.
        Each node in the graph must have a 'coordinates' attribute as an array of length D dimensions.
    ndim : int
        Number of dimensions for graph. When data is not None, ndim must be D.
        An empty points layer can be instantiated with arbitrary ndim.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    axis_labels : tuple of str, optional
        Dimension names of the layer data.
        If not provided, axis_labels will be set to (..., '-2', '-1').
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', 'translucent_no_depth', 'additive', 'minimum',
        and 'multiplicative'}.
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.
    metadata : dict
        Layer metadata.
    name : str
        Name of the layer. If not provided then will be guessed using heuristics.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    projection_mode : str
        How data outside the viewed dimensions but inside the thick Dims slice will
        be projected onto the viewed dimensions. Must fit to cls._projectionclass.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    scale : tuple of float
        Scale factors for the layer.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    translate : tuple of float
        Translation values for the layer.
    units : tuple of str or pint.Unit, optional
        Units of the layer data in world coordinates.
        If not provided, the default units are assumed to be pixels.
    visible : bool
        Whether the layer visual is currently being displayed.
    """

    _data: nxGraph

    def __init__(
        self,
        data=None,
        ndim=None,
        *,
        affine=None,
        axis_labels=None,
        blending='translucent',
        cache=True,
        experimental_clipping_planes=None,
        metadata=None,
        name=None,
        opacity=1.0,
        projection_mode=None,
        rotate=None,
        scale=None,
        shear=None,
        translate=None,
        units=None,
        visible=True,
    ) -> None:

        data, ndim = fix_graph_data(data, ndim)

        self._data = data
        super().__init__(
            data=data,
            ndim=ndim,
            affine=affine,
            axis_labels=axis_labels,
            blending=blending,
            cache=cache,
            experimental_clipping_planes=experimental_clipping_planes,
            metadata=metadata,
            name=name,
            opacity=opacity,
            projection_mode=projection_mode,
            rotate=rotate,
            scale=scale,
            shear=shear,
            translate=translate,
            units=units,
            visible=visible,
        )
        self.set_view_slice()

    @property
    def data(self) -> nxGraph:
        return self._data

    @data.setter
    def data(self, data: nxGraph) -> None:
        self._data = data
        self._update_dims()

    def _get_ndim(self) -> int:
        nodes = self.data.nodes(data=True)
        first_node = next(iter(nodes), None)

        return (
            self._ndim
            if first_node is None
            else len(first_node[1]['coordinates'])
        )

    @property
    def _extent_data(self) -> np.ndarray:
        if not self.data.nodes:
            return np.full((2, self.ndim), np.nan)
        coordinates = np.asarray(
            [
                node_attr['coordinates']
                for _, node_attr in self.data.nodes(data=True)
            ]
        )
        min_coords = np.min(coordinates, axis=0)
        max_coords = np.max(coordinates, axis=0)

        return np.vstack([min_coords, max_coords])

    def _get_layer_slicing_state(self, data, cache) -> _GraphSlicingState:
        return _GraphSlicingState(self, data, cache)

    def _get_state(self) -> dict:
        state = self._get_base_state()
        state.update({'data': self.data, 'ndim': self.ndim})
        return state

    def _set_view_slice(self) -> None:
        pass

    def _update_thumbnail(self) -> None:
        thumbnail = np.zeros(self._thumbnail_shape)
        thumbnail[..., 3] = self.opacity
        self.thumbnail = thumbnail

    def _get_value(self, position):
        return None
