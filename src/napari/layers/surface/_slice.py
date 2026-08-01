import warnings
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from napari.layers.base._slice import _next_request_id
from napari.layers.surface._surface_constants import SurfaceProjectionMode
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice

OptArray = npt.NDArray | None


@dataclass(frozen=True)
class _SurfaceSliceResponse:
    """Contains all the output data of slicing a surface layer.

    Attributes
    ----------
    vertices : array like
        Sliced vertices coordinates.
    faces : array like
        Faces reindexed for the sliced vertices.
    slice_input : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    request_id : int
        The identifier of the request from which this was generated.
    """

    vertices: np.ndarray = field(repr=False)
    faces: np.ndarray = field(repr=False)
    values: np.ndarray | None = field(repr=False)
    vertex_colors: np.ndarray | None = field(repr=False)
    texcoords: np.ndarray | None = field(repr=False)
    slice_input: _SliceInput
    request_id: int


@dataclass(frozen=True)
class _SurfaceSliceRequest:
    """A callable that stores all the input data needed to slice a Surface layer.

    This should be treated a deeply immutable structure, even though some
    fields can be modified in place. It is like a function that has captured
    all its inputs already.

    In general, the calling an instance of this may take a long time, so you may
    want to run it off the main thread.

    Attributes
    ----------
    slice_input : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    data : Any
        The layer's data field, which is the main input to slicing.
    data_slice : _ThickNDSlice
        The slicing coordinates and margins in data space.
    others
        See the corresponding attributes in `Layer` and `Points`.
    """

    slice_input: _SliceInput
    data: tuple = field(repr=False)
    vertex_colors: np.ndarray | None = field(repr=False)
    texcoords: np.ndarray | None = field(repr=False)
    data_slice: _ThickNDSlice = field(repr=False)
    projection_mode: SurfaceProjectionMode
    id: int = field(default_factory=_next_request_id)

    def __call__(self) -> _SurfaceSliceResponse:
        # Return early if no data
        if len(self.data) == 0:
            return self._empty_response()

        not_disp = list(self.slice_input.not_displayed)
        if not not_disp:
            return _SurfaceSliceResponse(
                vertices=self.data[0],
                faces=self.data[1],
                values=self.data[2] if len(self.data) == 3 else None,
                vertex_colors=self.vertex_colors,
                texcoords=self.texcoords,
                slice_input=self.slice_input,
                request_id=self.id,
            )

        # do the slicing based on the point and margins
        point, m_left, m_right = self.data_slice.as_array()

        if self.projection_mode == 'none':
            low = point.copy()
            high = point.copy()
        else:
            low = point - m_left
            high = point + m_right

        # assume slice thickness of 1 in data pixels
        # (same as before thick slices were implemented)
        too_thin_slice = np.isclose(high, low)
        low[too_thin_slice] -= 0.5
        high[too_thin_slice] += 0.5

        vert_orig = self.data[0]
        disp = list(self.slice_input.displayed)
        # the presence of vertex_values can add extra dimensions to the layer
        # which are not present in the data itself. We need to adjust for this
        # by offsetting the dim number by the amount of extra dims
        values_orig = self.data[2]
        extra_dims = values_orig.ndim - 1
        if any(d < extra_dims for d in disp):
            warnings.warn(
                'All extra dimensions corresponding to vertex values must be non-displayed dimensions. '
                'Data cannot be shown.',
                UserWarning,
                stacklevel=2,
            )
            return self._empty_response()

        disp_for_vert = [d - extra_dims for d in disp if d >= extra_dims]
        not_disp_for_vert = [
            d - extra_dims for d in not_disp if d >= extra_dims
        ]

        # actually do the slicing (for vertices)
        low_vert = low[range(extra_dims, vert_orig.shape[1])][
            not_disp_for_vert
        ]
        high_vert = high[range(extra_dims, vert_orig.shape[1])][
            not_disp_for_vert
        ]

        vertices_not_disp = vert_orig[:, not_disp_for_vert]
        inside_slice = np.all(
            (vertices_not_disp >= low_vert) & (vertices_not_disp <= high_vert),
            axis=1,
        )
        valid_vertices = np.argwhere(inside_slice).reshape(-1)

        vertices_disp = vert_orig[:, disp_for_vert]
        vertices = vertices_disp[valid_vertices]

        # mapping of old vertex indices to new vertex indices. Indexing at
        # a non-valid index is undefined, but shouldn't happen
        old_to_new = np.empty(vertices_not_disp.shape[0], dtype=int)
        old_to_new[valid_vertices] = np.arange(valid_vertices.shape[0])

        valid_mask = np.zeros(vertices_not_disp.shape[0], dtype=bool)
        valid_mask[valid_vertices] = True
        valid_faces_mask = valid_mask[self.data[1]].all(axis=1)
        valid_faces = self.data[1][valid_faces_mask]
        faces = old_to_new[valid_faces]

        values = vertex_colors = texcoords = None

        # we can only get here if all the values/colors extra dimensions are not displayed
        # which simplifies the logic (order still matters). We still need to remove non-displayed
        # dimensions that relate do the vertices and not to the values/colors.
        not_disp_for_values = [d for d in not_disp if d < extra_dims]
        slice_values = tuple(point[not_disp_for_values].astype(int))

        if self.vertex_colors is not None:
            vertex_colors = self.vertex_colors[slice_values][valid_vertices]
        elif len(self.data) == 3:
            values = values_orig[slice_values][valid_vertices]

        if self.texcoords is not None:
            texcoords = self.texcoords[valid_vertices]

        return _SurfaceSliceResponse(
            vertices=vertices,
            faces=faces,
            values=values,
            vertex_colors=vertex_colors,
            texcoords=texcoords,
            slice_input=self.slice_input,
            request_id=self.id,
        )

    def _empty_response(self) -> _SurfaceSliceResponse:
        return _SurfaceSliceResponse(
            vertices=np.empty((0, self.slice_input.ndisplay), dtype=int),
            faces=np.empty((0, 3), dtype=int),
            values=None,
            vertex_colors=None,
            texcoords=None,
            slice_input=self.slice_input,
            request_id=self.id,
        )
