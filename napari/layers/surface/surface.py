import warnings
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from napari.layers.base import Layer
from napari.layers.intensity_mixin import IntensityVisualizationMixin
from napari.layers.surface._surface_constants import Shading
from napari.layers.surface._surface_utils import (
    calculate_barycentric_coordinates,
)
from napari.layers.surface.normals import SurfaceNormals
from napari.layers.surface.wireframe import SurfaceWireframe
from napari.layers.utils.interactivity_utils import (
    nd_line_segment_to_displayed_data_ray,
)
from napari.layers.utils.layer_utils import calc_data_range
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.events import Event
from napari.utils.events.event_utils import connect_no_arg
from napari.utils.geometry import find_nearest_triangle_intersection
from napari.utils.translations import trans


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
    colormap : str, napari.utils.Colormap, tuple, dict
        Colormap to use for luminance images. If a string must be the name
        of a supported colormap from vispy or matplotlib. If a tuple the
        first value must be a string to assign as a name to a colormap and
        the second item must be a Colormap. If a dict the key must be a
        string to assign as a name to a colormap and the value must be a
        Colormap.
    texture: (I, J) or (I, J, C) array
        A 2D texture to be mapped onto the mesh using `texcoords`.
        C may be 3 (RGB) or 4 (RGBA) channels for a color texture.
    texcoords: (N, 2) array
        2D coordinates for each vertex, mapping into the texture.
        The number of texture coords must match the number of vertices (N).
        Coordinates should be in [0.0, 1.0] and are scaled to sample the 2D
        texture. Coordinates outside this range will wrap, but this behavior
        should be considered an implementation detail: there are no plans to
        change it, but it's a feature of the underlying vispy visual.
    vertex_colors: (N, C) or (K0, ..., KL, N, C) array of color values
        Take care that the (optional) L additional dimensions match those of
        vertex_values for proper slicing.
        C may be 3 (RGB) or 4 (RGBA) channels.
    contrast_limits : list (2,)
        Color limits to be used for determining the colormap bounds for
        luminance images. If not passed is calculated as the min and max of
        the image.
    gamma : float
        Gamma correction for determining colormap linearity. Defaults to 1.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    shading : str, Shading
        One of a list of preset shading modes that determine the lighting model
        using when rendering the surface in 3D.

        * ``Shading.NONE``
          Corresponds to ``shading='none'``.
        * ``Shading.FLAT``
          Corresponds to ``shading='flat'``.
        * ``Shading.SMOOTH``
          Corresponds to ``shading='smooth'``.
    visible : bool
        Whether the layer visual is currently being displayed.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    wireframe : None, dict or SurfaceWireframe
        Whether and how to display the edges of the surface mesh with a wireframe.
    normals : None, dict or SurfaceNormals
        Whether and how to display the face and vertex normals of the surface mesh.

    Attributes
    ----------
    data : 3-tuple of array
        The first element of the tuple is an (N, D) array of vertices of
        mesh triangles. The second is an (M, 3) array of int of indices
        of the mesh triangles. The third element is the (K0, ..., KL, N)
        array of values used to color vertices where the additional L
        dimensions are used to color the same mesh with different values.
    vertices : (N, D) array
        Vertices of mesh triangles.
    faces : (M, 3) array of int
        Indices of mesh triangles.
    vertex_values : (K0, ..., KL, N) array
        Values used to color vertices.
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


    Notes
    -----
    _data_view : (M, 2) or (M, 3) array
        The coordinates of the vertices given the viewed dimensions.
    _view_faces : (P, 3) array
        The integer indices of the vertices that form the triangles
        in the currently viewed slice.
    _colorbar : array
        Colorbar for current colormap.
    """

    _colormaps = AVAILABLE_COLORMAPS

    def __init__(
        self,
        data,
        *,
        colormap='gray',
        contrast_limits=None,
        gamma=1,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1,
        blending='translucent',
        shading='flat',
        visible=True,
        cache=True,
        experimental_clipping_planes=None,
        wireframe=None,
        normals=None,
        texture=None,
        texcoords=None,
        vertex_colors=None,
    ) -> None:
        ndim = data[0].shape[1]

        super().__init__(
            data,
            ndim,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            rotate=rotate,
            shear=shear,
            affine=affine,
            opacity=opacity,
            blending=blending,
            visible=visible,
            cache=cache,
            experimental_clipping_planes=experimental_clipping_planes,
        )

        self.events.add(
            interpolation=Event,
            rendering=Event,
            shading=Event,
            wireframe=Event,
            normals=Event,
            texture=Event,
            texcoords=Event,
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

        self._texture = texture
        self._texcoords = texcoords
        self._vertex_colors = vertex_colors

        # Set contrast_limits and colormaps
        self._gamma = gamma
        if contrast_limits is None:
            self._contrast_limits_range = calc_data_range(self._vertex_values)
        else:
            self._contrast_limits_range = contrast_limits
        self._contrast_limits = tuple(self._contrast_limits_range)
        self.colormap = colormap
        self.contrast_limits = self._contrast_limits

        # Data containing vectors in the currently viewed slice
        self._data_view = np.zeros((0, self._slice_input.ndisplay))
        self._view_faces = np.zeros((0, 3))
        self._view_vertex_values = []
        self._view_vertex_colors = []

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

    def _calc_data_range(self, mode='data'):
        return calc_data_range(self.vertex_values)

    @property
    def dtype(self):
        return self.vertex_values.dtype

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
    def vertex_values(self, vertex_values: np.ndarray):
        """Array of values (n, 1) used to color vertices with a colormap."""
        if vertex_values is None:
            vertex_values = np.ones(len(self._vertices))

        self._vertex_values = vertex_values

        self._update_dims()
        self.events.data(value=self.data)
        self._reset_editable()

    @property
    def vertex_colors(self) -> Optional[np.ndarray]:
        return self._vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, vertex_colors: Optional[np.ndarray]):
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
                f"texture should be None or ndarray; got {type(vertex_colors)}"
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
    def faces(self, faces: np.ndarray):
        """Array of indices of mesh triangles."""

        self.faces = faces

        self.refresh()
        self.events.data(value=self.data)
        self._reset_editable()

    def _get_ndim(self):
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
    def shading(self):
        return str(self._shading)

    @shading.setter
    def shading(self, shading):
        if isinstance(shading, Shading):
            self._shading = shading
        else:
            self._shading = Shading(shading)
        self.events.shading(value=self._shading)

    @property
    def wireframe(self) -> SurfaceWireframe:
        return self._wireframe

    @wireframe.setter
    def wireframe(self, wireframe: Union[dict, SurfaceWireframe, None]):
        if wireframe is None:
            self._wireframe.reset()
        elif isinstance(wireframe, (SurfaceWireframe, dict)):
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
    def normals(self, normals: Union[dict, SurfaceNormals, None]):
        if normals is None:
            self._normals.reset()
        elif not isinstance(normals, (SurfaceNormals, dict)):
            raise ValueError(
                f'normals should be None, a dict, or SurfaceNormals; got {type(normals)}'
            )
        else:
            if isinstance(normals, SurfaceNormals):
                normals = {k: dict(v) for k, v in normals.dict().items()}
            # ignore modes, they are unmutable cause errors
            for norm_type in ('face', 'vertex'):
                normals.get(norm_type, {}).pop('mode', None)
            self._normals.update(normals)
        self.events.normals(value=self._normals)

    @property
    def texture(self) -> Optional[np.ndarray]:
        return self._texture

    @texture.setter
    def texture(self, texture: np.ndarray):
        if texture is not None and not isinstance(texture, np.ndarray):
            msg = f"texture should be None or ndarray; got {type(texture)}"
            raise ValueError(msg)
        self._texture = texture
        self.events.texture(value=self._texture)

    @property
    def texcoords(self) -> Optional[np.ndarray]:
        return self._texcoords

    @texcoords.setter
    def texcoords(self, texcoords: np.ndarray):
        if texcoords is not None and not isinstance(texcoords, np.ndarray):
            msg = f"texcoords should be None or ndarray; got {type(texcoords)}"
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
        )

    def _get_state(self):
        """Get dictionary of layer state.

        Returns
        -------
        state : dict
            Dictionary of layer state.
        """
        state = self._get_base_state()
        state.update(
            {
                'colormap': self.colormap.name,
                'contrast_limits': self.contrast_limits,
                'gamma': self.gamma,
                'shading': self.shading,
                'data': self.data,
                'wireframe': self.wireframe.dict(),
                'normals': self.normals.dict(),
                'texture': self.texture,
                'texcoords': self.texcoords,
                'vertex_colors': self.vertex_colors,
            }
        )
        return state

    def _slice_associated_data(
        self,
        data: np.ndarray,
        vertex_ndim: int,
        dims: int = 1,
    ) -> Union[List[Any], np.ndarray]:
        """Return associated layer data (e.g. vertex values, colors) within
        the current slice.
        """
        if data is None:
            return []

        data_ndim = data.ndim - 1
        if data_ndim >= dims:
            # Get indices for axes corresponding to data dimensions
            data_indices = self._slice_indices[:-vertex_ndim]
            data = data[data_indices]
            if data.ndim > dims:
                warnings.warn(
                    trans._(
                        "Assigning multiple data per vertex after slicing "
                        "is not allowed. All dimensions corresponding to "
                        "vertex data must be non-displayed dimensions. Data "
                        "may not be visible.",
                        deferred=True,
                    ),
                    category=UserWarning,
                    stacklevel=2,
                )
                return []
        return data

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""
        N, vertex_ndim = self.vertices.shape
        values_ndim = self.vertex_values.ndim - 1

        self._view_vertex_values = self._slice_associated_data(
            self.vertex_values,
            vertex_ndim,
        )

        self._view_vertex_colors = self._slice_associated_data(
            self.vertex_colors,
            vertex_ndim,
            dims=2,
        )

        if len(self._view_vertex_values) == 0:
            self._data_view = np.zeros((0, self._slice_input.ndisplay))
            self._view_faces = np.zeros((0, 3))
            return

        if values_ndim > 0:
            indices = np.array(self._slice_indices[-vertex_ndim:])
            disp = [
                d
                for d in np.subtract(self._slice_input.displayed, values_ndim)
                if d >= 0
            ]
            not_disp = [
                d
                for d in np.subtract(
                    self._slice_input.not_displayed, values_ndim
                )
                if d >= 0
            ]
        else:
            indices = np.array(self._slice_indices)
            not_disp = list(self._slice_input.not_displayed)
            disp = list(self._slice_input.displayed)

        self._data_view = self.vertices[:, disp]
        if len(self.vertices) == 0:
            self._view_faces = np.zeros((0, 3))
        elif vertex_ndim > self._slice_input.ndisplay:
            vertices = self.vertices[:, not_disp].astype('int')
            triangles = vertices[self.faces]
            matches = np.all(triangles == indices[not_disp], axis=(1, 2))
            matches = np.where(matches)[0]
            if len(matches) == 0:
                self._view_faces = np.zeros((0, 3))
            else:
                self._view_faces = self.faces[matches]
        else:
            self._view_faces = self.faces

        if self._keep_auto_contrast:
            self.reset_contrast_limits()

    def _update_thumbnail(self):
        """Update thumbnail with current surface."""

    def _get_value(self, position):
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
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Tuple[Union[None, float, int], Optional[int]]:
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
        mesh_triangles = self._data_view[self._view_faces]

        # get the triangles intersection
        intersection_index, intersection = find_nearest_triangle_intersection(
            ray_position=start_position,
            ray_direction=ray_direction,
            triangles=mesh_triangles,
        )

        if intersection_index is None:
            return None, None

        # add the full nD coords to intersection
        intersection_point = start_point.copy()
        intersection_point[dims_displayed] = intersection

        # calculate the value from the intersection
        triangle_vertex_indices = self._view_faces[intersection_index]
        triangle_vertices = self._data_view[triangle_vertex_indices]
        barycentric_coordinates = calculate_barycentric_coordinates(
            intersection, triangle_vertices
        )
        vertex_values = self._view_vertex_values[triangle_vertex_indices]
        intersection_value = (barycentric_coordinates * vertex_values).sum()

        return intersection_value, intersection_index
