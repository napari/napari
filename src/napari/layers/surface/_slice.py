from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from napari.layers.base._slice import _next_request_id
from napari.layers.surface._surface_constants import SurfaceProjectionMode
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice


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
    data_slice: _ThickNDSlice = field(repr=False)
    projection_mode: SurfaceProjectionMode
    id: int = field(default_factory=_next_request_id)

    def __call__(self) -> _SurfaceSliceResponse:
        # Return early if no data
        if len(self.data) == 0:
            return _SurfaceSliceResponse(
                vertices=np.empty((0, self.slice_input.ndisplay), dtype=int),
                faces=np.empty((0, 3), dtype=int),
                values=None,
                vertex_colors=None,
                slice_input=self.slice_input,
                request_id=self.id,
            )

        not_disp = list(self.slice_input.not_displayed)
        if not not_disp:
            # If we want to display everything, then use all indices.
            # scale is only impacted by not displayed data, therefore 1
            return _SurfaceSliceResponse(
                vertices=self.data[0],
                faces=self.data[1],
                values=self.data[2] if len(self.data) == 3 else None,
                vertex_colors=self.vertex_colors,
                slice_input=self.slice_input,
                request_id=self.id,
            )

        vertices, faces, values, vertex_colors = self._get_slice_data(not_disp)

        return _SurfaceSliceResponse(
            vertices=vertices,
            faces=faces,
            values=values,
            vertex_colors=vertex_colors,
            slice_input=self.slice_input,
            request_id=self.id,
        )

    def _get_slice_data(
        self, not_disp: list[int]
    ) -> tuple[npt.NDArray, npt.NDArray, Any, Any]:
        vertices = self.data[0][:, not_disp]
        faces = self.data[1]

        point, m_left, m_right = self.data_slice[not_disp].as_array()

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

        inside_slice = np.all((vertices >= low) & (vertices <= high), axis=1)

        valid_vertices = np.argwhere(inside_slice).squeeze()
        reindexed_vertices = vertices[valid_vertices]

        # mapping of old vertex indices to new vertex indices. Indexing at
        # a non-valid index is undefined, but shouldn't happen
        old_to_new = np.empty(len(vertices), dtype=int)
        old_to_new[valid_vertices] = np.arange(len(valid_vertices))

        valid_faces = np.all(np.isin(faces, valid_vertices), axis=1)
        reindexed_faces = old_to_new[valid_faces]

        return (
            reindexed_vertices,
            reindexed_faces,
            None,
            None,
        )
