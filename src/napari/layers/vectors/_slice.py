from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from napari.layers.base._slice import _next_request_id
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.layers.vectors._vectors_constants import VectorsProjectionMode


@dataclass(frozen=True)
class _VectorSliceResponse:
    """Contains all the output data of slicing an Vectors layer.

    Attributes
    ----------
    indices : array like
        Indices of the sliced Vectors data.
    alphas : array like or scalar
        Used to change the opacity of the sliced vectors for visualization.
        Should be broadcastable to indices.
    slice_input : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    request_id : int
        The identifier of the request from which this was generated.
    """

    indices: np.ndarray = field(repr=False)
    alphas: np.ndarray | float = field(repr=False)
    slice_input: _SliceInput
    request_id: int


@dataclass(frozen=True)
class _VectorSliceRequest:
    """A callable that stores all the input data needed to slice a Vectors layer.

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
        See the corresponding attributes in `Layer` and `Vectors`.
    """

    slice_input: _SliceInput
    data: Any = field(repr=False)
    data_slice: _ThickNDSlice = field(repr=False)
    projection_mode: VectorsProjectionMode
    length: float = field(repr=False)
    id: int = field(default_factory=_next_request_id)

    def __call__(self) -> _VectorSliceResponse:
        # Return early if no data
        if len(self.data) == 0:
            return _VectorSliceResponse(
                indices=np.empty(0, dtype=int),
                alphas=np.empty(0),
                slice_input=self.slice_input,
                request_id=self.id,
            )

        not_disp = list(self.slice_input.not_displayed)
        if not not_disp:
            # If we want to display everything, then use all indices.
            # alpha is only impacted by not displayed data, therefore 1
            return _VectorSliceResponse(
                indices=np.arange(len(self.data), dtype=int),
                alphas=np.ones(len(self.data)),
                slice_input=self.slice_input,
                request_id=self.id,
            )

        slice_indices, alphas = self._get_slice_data(not_disp)

        return _VectorSliceResponse(
            indices=slice_indices,
            alphas=alphas,
            slice_input=self.slice_input,
            request_id=self.id,
        )

    def _get_slice_data(
        self, not_disp: list[int]
    ) -> tuple[npt.NDArray, npt.NDArray | int]:

        point, m_left, m_right = self.data_slice[not_disp].as_array()

        if self.projection_mode == VectorsProjectionMode.NONE:
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

        coords_not_disp = self.data[:, 0, not_disp]
        inside_slice = np.all(
            (coords_not_disp >= low) & (coords_not_disp <= high), axis=1
        )
        visible = np.where(inside_slice)[0].astype(int)

        if not visible.size:
            return (
                np.empty(0, dtype=int),
                np.empty(0, dtype=float),
            )

        alphas = np.ones(len(visible))

        if self.projection_mode == VectorsProjectionMode.FADE:
            # rescale alphas of vectors based on how far they are from the center
            dist_from_point = coords_not_disp[visible] - point
            # margins can be different, so we need to treat low/high distance independently
            slice_end = np.where(
                dist_from_point < 0, low - point, high - point
            )
            # we multiply the alphas from each dimension into a single one
            alphas = np.prod(1 - dist_from_point / slice_end, axis=1)

        return visible, alphas
