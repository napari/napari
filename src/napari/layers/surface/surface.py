from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from napari.layers.base import Layer, _LayerSlicingState
from napari.layers.intensity_mixin import IntensityVisualizationMixin
from napari.layers.surface._slice import (
    _SurfaceSliceRequest,
    _SurfaceSliceResponse,
)
from napari.layers.surface._surface_constants import (
    Shading,
    SurfaceProjectionMode,
)
from napari.layers.surface._surface_utils import (
    calculate_barycentric_coordinates,
)
from napari.layers.surface.normals import SurfaceNormals
from napari.layers.surface.wireframe import SurfaceWireframe
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.layers.utils.interactivity_utils import (
    nd_line_segment_to_displayed_data_ray,
)
from napari.layers.utils.layer_utils import _FeatureTable, calc_data_range
from napari.types import LayerDataType
from napari.utils._dtype import normalize_dtype
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.events import Event
from napari.utils.events.event_utils import connect_no_arg
from napari.utils.geometry import find_nearest_triangle_intersection
from napari.utils.misc import StringEnum
from napari.utils.translations import trans

if TYPE_CHECKING:
    import pandas as pd

    from napari.components.dims import Dims


# Mixin must come before Layer
class Surface(IntensityVisualizationMixin, Layer):
    """
    Surface layer renders meshes onto the canvas.

    Surfaces may be colored by:
        * setting `vertex_values`, which colors the surface with the selected
          `colormap` (default is uniform ones)
        * setting `vertex_colors`, which replaces/overrides any color from
          `vertex_values`
        * setting both `texture` and `texcoords`, which blends a the value from
          a texture (image) with the underlying color from `vertex_values` or
          `vertex_colors`. Blending is achieved by multiplying the texture
          color by the underlying color - an underlying value of "white" will
          result in the unaltered texture color.

    Parameters
    ----------
    data : 2-tuple or 3-tuple of array
        The first element of the tuple is an (N, D) array of vertices of
        mesh triangles.

        The second is an (M, 3) array of int of indices of the mesh triangles.

        The optional third element is the (K0, ..., KL, N) array of values
        (vertex_values) used to color vertices where the additional L
        dimensions are used to color the same mesh with different values. If
        not provided, it defaults to ones.
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
        {'opaque', 'translucent', and 'additive'}.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    colormap : str, napari.utils.Colormap, tuple, dict
        Colormap to use for luminance images. If a string must be the name
        of a supported colormap from vispy or matplotlib. If a tuple the
        first value must be a string to assign as a name to a colormap and
        the second item must be a Colormap. If a dict the key must be a
        string to assign as a name to a colormap and the value must be a
        Colormap.
    contrast_limits : list (2,)
        Color limits to be used for determining the colormap bounds for
        luminance images. If not passed is calculated as the min and max of
        the image.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.
    feature_defaults : dict[str, Any] or Dataframe-like
        The default value of each feature in a table with one row.
    features : dict[str, array-like] or Dataframe-like
        Features table where each row corresponds to a shape and each column
        is a feature.
    gamma : float
        Gamma correction for determining colormap linearity. Defaults to 1.
    metadata : dict
        Layer metadata.
    name : str
        Name of the layer.
    normals : None, dict or SurfaceNormals
        Whether and how to display the face and vertex normals of the surface mesh.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    projection_mode : str
        How data outside the viewed dimensions but inside the thick Dims slice will
        be projected onto the viewed dimenions.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    scale : tuple of float
        Scale factors for the layer.
    shading : str, Shading
        One of a list of preset shading modes that determine the lighting model
        using when rendering the surface in 3D.

        * ``Shading.NONE``
          Corresponds to ``shading='none'``.
        * ``Shading.FLAT``
          Corresponds to ``shading='flat'``.
        * ``Shading.SMOOTH``
          Corresponds to ``shading='smooth'``.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    texcoords: (N, 2) array
        2D coordinates for each vertex, mapping into the texture.
        The number of texture coords must match the number of vertices (N).
        Coordinates should be in [0.0, 1.0] and are scaled to sample the 2D
        texture. Coordinates outside this range will wrap, but this behavior
        should be considered an implementation detail: there are no plans to
        change it, but it's a feature of the underlying vispy visual.
    texture: (I, J) or (I, J, C) array
        A 2D texture to be mapped onto the mesh using `texcoords`.
        C may be 3 (RGB) or 4 (RGBA) channels for a color texture.
    translate : tuple of float
        Translation values for the layer
    units : tuple of str or pint.Unit, optional
        Units of the layer data in world coordinates.
        If not provided, the default units are assumed to be pixels.
    vertex_colors: (N, C) or (K0, ..., KL, N, C) array of color values
        Take care that the (optional) L additional dimensions match those of
        vertex_values for proper slicing.
        C may be 3 (RGB) or 4 (RGBA) channels..
    visible : bool
        Whether the layer visual is currently being displayed.
    wireframe : None, dict or SurfaceWireframe
        Whether and how to display the edges of the surface mesh with a wireframe.


    Attributes
    ----------
    data : 3-tuple of array
        The first element of the tuple is an (N, D) array of vertices of
        mesh triangles. The second is an (M, 3) array of int of indices
        of the mesh triangles. The third element is the (K0, ..., KL, N)
        array of values used to color vertices where the additional L
        dimensions are used to color the same mesh with different values.
    axis_labels : tuple of str
        Dimension names of the layer data.
    vertices : (N, D) array
        Vertices of mesh triangles.
    faces : (M, 3) array of int
        Indices of mesh triangles.
    vertex_values : (K0, ..., KL, N) array
        Values used to color vertices.
    features : DataFrame-like
        Features table where each row corresponds to a vertex and each column
        is a feature.
    feature_defaults : DataFrame-like
        Stores the default value of each feature in a table with one row.
    colormap : str, napari.utils.Colormap, tuple, dict
        Colormap to use for luminance images. If a string must be the name
        of a supported colormap from vispy or matplotlib. If a tuple the
        first value must be a string to assign as a name to a colormap and
        the second item must be a Colormap. If a dict the key must be a
        string to assign as a name to a colormap and the value must be a
        Colormap.
    contrast_limits : list (2,)
        Color limits to be used for determining the colormap bounds for
        luminance images. If not passed is calculated as the min and max of
        the image.
    shading: str
        One of a list of preset shading modes that determine the lighting model
        using when rendering the surface.

        * ``'none'``
        * ``'flat'``
        * ``'smooth'``
    gamma : float
        Gamma correction for determining colormap linearity.
    wireframe : SurfaceWireframe
        Whether and how to display the edges of the surface mesh with a wireframe.
    normals : SurfaceNormals
        Whether and how to display the face and vertex normals of the surface mesh.
    units: tuple of pint.Unit
        Units of the layer data in world coordinates.


    Notes
    -----
    _view_vertices : (M, 2) or (M, 3) array
        The coordinates of the vertices given the viewed dimensions.
    _view_faces : (P, 3) array
        The integer indices of the vertices that form the triangles
        in the currently viewed slice.
    _colorbar : array
        Colorbar for current colormap.
    """

    _projectionclass: type[StringEnum] = SurfaceProjectionMode

    _colormaps = AVAILABLE_COLORMAPS

    def __init__(
        self,
        data,
        *,
        affine=None,
        axis_labels=None,
        blending='translucent',
        cache=True,
        colormap='gray',
        contrast_limits=None,
        experimental_clipping_planes=None,
        feature_defaults=None,
        features=None,
        gamma=1.0,
        metadata=None,
        name=None,
        normals=None,
        opacity=1.0,
        projection_mode='all',
        rotate=None,
        scale=None,
        shading='flat',
        shear=None,
        texcoords=None,
        texture=None,
        translate=None,
        units=None,
        vertex_colors=None,
        visible=True,
        wireframe=None,
    ) -> None:
        ndim = data[0].shape[1]

        super().__init__(
            data,
            ndim,
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

        self.events.add(
            interpolation=Event,
            rendering=Event,
            shading=Event,
            wireframe=Event,
            normals=Event,
            texture=Event,
            texcoords=Event,
            features=Event,
            feature_defaults=Event,
        )

        # assign mesh data and establish default behavior
        if len(data) not in (2, 3):
            raise ValueError(
                trans._(
                    'Surface data tuple must be 2 or 3, specifying vertices, faces, and optionally vertex values, instead got length {length}.',
                    deferred=True,
                    length=len(data),
                )
            )
        self._vertices = data[0]
        self._faces = data[1]
        if len(data) == 3:
            self._vertex_values = data[2]
        else:
            self._vertex_values = np.ones(len(self._vertices))

        self._feature_table = _FeatureTable.from_layer(
            features=features,
            feature_defaults=feature_defaults,
            num_data=len(data[0]),
        )

        self._texture = texture
        self._texcoords = texcoords
        self._vertex_colors = vertex_colors

        # Set contrast_limits and colormaps
        self._gamma = gamma
        if contrast_limits is not None:
            self._contrast_limits_range = contrast_limits
        else:
            self._contrast_limits_range = calc_data_range(self._vertex_values)

        self._contrast_limits = self._contrast_limits_range
        self.colormap = colormap
        self.contrast_limits = self._contrast_limits

        # Trigger generation of view slice and thumbnail.
        # Use _update_dims instead of refresh here because _get_ndim is
        # dependent on vertex_values as well as vertices.
        self._update_dims()

        # Shading mode
        self._shading = shading

        # initialize normals and wireframe
        self._wireframe = SurfaceWireframe()
        self._normals = SurfaceNormals()
        connect_no_arg(self.wireframe.events, self.events, 'wireframe')
        connect_no_arg(self.normals.events, self.events, 'normals')

        self.wireframe = wireframe
        self.normals = normals
        self._slicing_state.slice_done.connect(
            self._maybe_reset_contrast_limits
        )

    @property
    def _view_vertex_colors(self) -> list[Any] | np.ndarray:
        return self._slicing_state._view_vertex_colors

    @property
    def _view_vertex_values(self) -> list[Any] | np.ndarray:
        return self._slicing_state._view_vertex_values

    @property
    def _view_vertices(self) -> np.ndarray:
        return self._slicing_state._view_vertices

    @property
    def _view_faces(self) -> np.ndarray:
        return self._slicing_state._view_faces

    @property
    def _view_texcoords(self) -> np.ndarray | None:
        return self._slicing_state._view_texcoords

    def _calc_data_range(self, mode='data'):
        return calc_data_range(self.vertex_values)

    @property
    def dtype(self) -> np.dtype:
        return normalize_dtype(self.vertex_values.dtype)

    @property
    def data(self):
        return (self.vertices, self.faces, self.vertex_values)

    @data.setter
    def data(self, data):
        if len(data) not in (2, 3):
            raise ValueError(
                trans._(
                    'Surface data tuple must be 2 or 3, specifying vertices, faces, and optionally vertex values, instead got length {data_length}.',
                    deferred=True,
                    data_length=len(data),
                )
            )
        self._vertices = data[0]
        self._faces = data[1]
        if len(data) == 3:
            self._vertex_values = data[2]
        else:
            self._vertex_values = np.ones(len(self._vertices))

        self._update_dims()
        self.events.data(value=self.data)
        self._reset_editable()
        if self._keep_auto_contrast:
            self.reset_contrast_limits()

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        """Array of vertices of mesh triangles."""

        self._vertices = vertices

        self._update_dims()
        self.events.data(value=self.data)
        self._reset_editable()

    @property
    def vertex_values(self) -> np.ndarray:
        return self._vertex_values

    @vertex_values.setter
    def vertex_values(self, vertex_values: np.ndarray) -> None:
        """Array of values (n, 1) used to color vertices with a colormap."""
        if vertex_values is None:
            vertex_values = np.ones(len(self._vertices))

        self._vertex_values = vertex_values

        self._update_dims()
        self.events.data(value=self.data)
        self._reset_editable()

    @property
    def vertex_colors(self) -> np.ndarray | None:
        return self._vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, vertex_colors: np.ndarray | None) -> None:
        """Values used to directly color vertices.

        Note that dims sliders for this layer are based on vertex_values, so
        make sure the shape of vertex_colors matches the shape of vertex_values
        for proper slicing. That is: vertex_colors should be None, one set
        (N, C), or completely match the dimensions of vertex_values
        (K0, ..., KL, N, C).
        """
        if vertex_colors is not None and not isinstance(
            vertex_colors, np.ndarray
        ):
            msg = (
                f'texture should be None or ndarray; got {type(vertex_colors)}'
            )
            raise ValueError(msg)
        self._vertex_colors = vertex_colors
        self._update_dims()
        self.events.data(value=self.data)
        self._reset_editable()

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    @faces.setter
    def faces(self, faces: np.ndarray) -> None:
        """Array of indices of mesh triangles."""

        self.faces = faces

        self.refresh(extent=False)
        self.events.data(value=self.data)
        self._reset_editable()

    def _get_ndim(self) -> int:
        """Determine number of dimensions of the layer."""
        return self.vertices.shape[1] + (self.vertex_values.ndim - 1)

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        if len(self.vertices) == 0:
            extrema = np.full((2, self.ndim), np.nan)
        else:
            maxs = np.max(self.vertices, axis=0)
            mins = np.min(self.vertices, axis=0)

            # The full dimensionality and shape of the layer is determined by
            # the number of additional vertex value dimensions and the
            # dimensionality of the vertices themselves
            if self.vertex_values.ndim > 1:
                mins = [0] * (self.vertex_values.ndim - 1) + list(mins)
                maxs = [n - 1 for n in self.vertex_values.shape[:-1]] + list(
                    maxs
                )
            extrema = np.vstack([mins, maxs])
        return extrema

    @property
    def features(self) -> pd.DataFrame:
        """Dataframe-like features table.

        It is an implementation detail that this is a `pandas.DataFrame`. In the future,
        we will target the currently-in-development Data API dataframe protocol [1].
        This will enable us to use alternate libraries such as xarray or cuDF for
        additional features without breaking existing usage of this.

        If you need to specifically rely on the pandas API, please coerce this to a
        `pandas.DataFrame` using `features_to_pandas_dataframe`.

        References
        ----------
        .. [1]: https://data-apis.org/dataframe-protocol/latest/API.html
        """
        return self._feature_table.values

    @features.setter
    def features(
        self,
        features: dict[str, np.ndarray] | pd.DataFrame,
    ) -> None:
        self._feature_table.set_values(features, num_data=len(self.data[0]))
        self.events.features()

    @property
    def feature_defaults(self) -> pd.DataFrame:
        """Dataframe-like with one row of feature default values.

        See `features` for more details on the type of this property.
        """
        return self._feature_table.defaults

    @feature_defaults.setter
    def feature_defaults(
        self, defaults: dict[str, Any] | pd.DataFrame
    ) -> None:
        self._feature_table.set_defaults(defaults)
        self.events.feature_defaults()

    @property
    def shading(self) -> str:
        return str(self._shading)

    @shading.setter
    def shading(self, shading: str | Shading) -> None:
        if isinstance(shading, Shading):
            self._shading = shading
        else:
            self._shading = Shading(shading)
        self.events.shading(value=self._shading)

    @property
    def wireframe(self) -> SurfaceWireframe:
        return self._wireframe

    @wireframe.setter
    def wireframe(self, wireframe: dict | SurfaceWireframe | None) -> None:
        if wireframe is None:
            self._wireframe.reset()
        elif isinstance(wireframe, SurfaceWireframe | dict):
            self._wireframe.update(wireframe)
        else:
            raise ValueError(
                f'wireframe should be None, a dict, or SurfaceWireframe; got {type(wireframe)}'
            )
        self.events.wireframe(value=self._wireframe)

    @property
    def normals(self) -> SurfaceNormals:
        return self._normals

    @normals.setter
    def normals(self, normals: dict | SurfaceNormals | None) -> None:
        if normals is None:
            self._normals.reset()
        elif not isinstance(normals, SurfaceNormals | dict):
            raise ValueError(
                f'normals should be None, a dict, or SurfaceNormals; got {type(normals)}'
            )
        else:
            if isinstance(normals, SurfaceNormals):
                normals = {k: dict(v) for k, v in normals.model_dump().items()}
            # ignore modes, they are unmutable cause errors
            for norm_type in ('face', 'vertex'):
                normals.get(norm_type, {}).pop('mode', None)
            self._normals.update(normals)
        self.events.normals(value=self._normals)

    @property
    def texture(self) -> np.ndarray | None:
        return self._texture

    @texture.setter
    def texture(self, texture: np.ndarray) -> None:
        if texture is not None and not isinstance(texture, np.ndarray):
            msg = f'texture should be None or ndarray; got {type(texture)}'
            raise ValueError(msg)
        self._texture = texture
        self.events.texture(value=self._texture)

    @property
    def texcoords(self) -> np.ndarray | None:
        return self._texcoords

    @texcoords.setter
    def texcoords(self, texcoords: np.ndarray) -> None:
        if texcoords is not None and not isinstance(texcoords, np.ndarray):
            msg = f'texcoords should be None or ndarray; got {type(texcoords)}'
            raise ValueError(msg)
        self._texcoords = texcoords
        self.events.texcoords(value=self._texcoords)

    @property
    def _has_texture(self) -> bool:
        """Whether the layer has sufficient data for texturing"""
        return bool(
            self.texture is not None
            and self.texcoords is not None
            and len(self.texcoords)
            and self._view_texcoords is not None
        )

    def _get_state(self) -> dict[str, Any]:
        """Get dictionary of layer state.

        Returns
        -------
        state : dict of str to Any
            Dictionary of layer state.
        """
        state = self._get_base_state()
        state.update(
            {
                'colormap': self.colormap.model_dump(),
                'contrast_limits': self.contrast_limits,
                'gamma': self.gamma,
                'shading': self.shading,
                'data': self.data,
                'features': self.features,
                'feature_defaults': self.feature_defaults,
                'wireframe': self.wireframe.model_dump(),
                'normals': self.normals.model_dump(),
                'texture': self.texture,
                'texcoords': self.texcoords,
                'vertex_colors': self.vertex_colors,
            }
        )
        return state

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""
        raise NotImplementedError

    def _update_thumbnail(self) -> None:
        """Update thumbnail with current surface."""

    def _get_value(self, position) -> None:
        """Value of the data at a position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : None
            Value of the data at the coord.
        """
        return

    def _get_value_3d(
        self,
        start_point: np.ndarray | None,
        end_point: np.ndarray | None,
        dims_displayed: list[int],
    ) -> tuple[None | float | int, int | None]:
        """Get the layer data value along a ray

        Parameters
        ----------
        start_point : np.ndarray
            The start position of the ray used to interrogate the data.
        end_point : np.ndarray
            The end position of the ray used to interrogate the data.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the Viewer.

        Returns
        -------
        value
            The data value along the supplied ray.
        vertex : None
            Index of vertex if any that is at the coordinates.
        """
        if len(dims_displayed) != 3:
            # only applies to 3D
            return None, None
        if (start_point is None) or (end_point is None):
            # return None if the ray doesn't intersect the data bounding box
            return None, None

        start_position, ray_direction = nd_line_segment_to_displayed_data_ray(
            start_point=start_point,
            end_point=end_point,
            dims_displayed=dims_displayed,
        )

        # get the mesh triangles
        mesh_triangles = self._view_vertices[self._view_faces]

        # get the triangles intersection
        intersection_index, intersection = find_nearest_triangle_intersection(
            ray_position=start_position,
            ray_direction=ray_direction,
            triangles=mesh_triangles,
        )

        if (
            intersection_index is None
            or intersection is None
            or self._view_vertex_values is None
        ):
            return None, None

        # add the full nD coords to intersection
        intersection_point = start_point.copy()
        intersection_point[dims_displayed] = intersection

        # calculate the value from the intersection
        triangle_vertex_indices = self._view_faces[intersection_index]
        triangle_vertices = self._view_vertices[triangle_vertex_indices]
        barycentric_coordinates = calculate_barycentric_coordinates(
            intersection, triangle_vertices
        )
        vertex_values = self._view_vertex_values[triangle_vertex_indices]
        intersection_value = (barycentric_coordinates * vertex_values).sum()

        return intersection_value, intersection_index

    def __copy__(self):
        """Create a copy of this layer.

        Returns
        -------
        layer : napari.layers.Layer
            Copy of this layer.

        Notes
        -----
        This method is defined for purpose of asv memory benchmarks.
        The copy of data is intentional for properly estimating memory
        usage for layer.

        If you want a to copy a layer without coping the data please use
        `layer.create(*layer.as_layer_data_tuple())`

        If you change this method, validate if memory benchmarks are still
        working properly.
        """
        _data, meta, layer_type = self.as_layer_data_tuple()
        return self.create(
            tuple(copy.copy(x) for x in self.data),
            meta=meta,
            layer_type=layer_type,
        )

    def _get_layer_slicing_state(
        self, data: LayerDataType, cache: bool
    ) -> _SurfaceSlicingState:
        return _SurfaceSlicingState(layer=self, data=data, cache=cache)

    def _maybe_reset_contrast_limits(self) -> None:
        if self._keep_auto_contrast:
            self.reset_contrast_limits()


class _SurfaceSlicingState(_LayerSlicingState):
    layer: Surface

    def __init__(self, layer: Surface, data, cache: bool):
        super().__init__(layer=layer, data=data, cache=cache)
        # Data containing vectors in the currently viewed slice
        self._view_vertices = np.zeros((0, self._slice_input.ndisplay))
        self._view_faces = np.zeros((0, 3), dtype=int)
        self._view_vertex_values: np.ndarray | None = None
        self._view_vertex_colors: np.ndarray | None = None
        self._view_texcoords: np.ndarray | None = None

    def _set_view_slice(self) -> None:
        """Sets the view given the indices to slice with."""

        # The new slicing code makes a request from the existing state and
        # executes the request on the calling thread directly.
        # For async slicing, the calling thread will not be the main thread.
        request = self.make_slice_request_internal(
            self._slice_input, self.data_slice
        )
        response = request()
        self._update_slice_response(response)

    def _make_slice_request(self, dims: Dims) -> _SurfaceSliceRequest:
        """Make a Surface slice request based on the given dims and these data."""
        slice_input = self.make_slice_input(dims)
        # See Image._make_slice_request to understand why we evaluate this here
        # instead of using `self._data_slice`.
        data_slice = self._slice_indices(slice_input, dims)
        return self.make_slice_request_internal(slice_input, data_slice)

    def make_slice_request_internal(
        self, slice_input: _SliceInput, data_slice: _ThickNDSlice
    ) -> _SurfaceSliceRequest:
        return _SurfaceSliceRequest(
            slice_input=slice_input,
            data=self.layer.data,
            vertex_colors=self.layer.vertex_colors,
            texcoords=self.layer.texcoords,
            data_slice=data_slice,
            projection_mode=self.layer.projection_mode,
        )

    def _update_slice_response(self, response: _SurfaceSliceResponse) -> None:
        """Handle a slicing response."""
        self._slice_input = response.slice_input
        self._view_vertices = response.vertices
        self._view_faces = response.faces
        self._view_vertex_values = response.values
        self._view_vertex_colors = response.vertex_colors
        self._view_texcoords = response.texcoords
