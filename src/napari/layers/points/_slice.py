from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from napari.layers.base._slice import _next_request_id
from napari.layers.points._points_constants import PointsProjectionMode
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice


@dataclass(frozen=True)
class _PointSliceResponse:
    """Contains all the output data of slicing an points layer.

    Attributes
    ----------
    indices : array like
        Indices of the visible points.
    size: array like
        Sizes of the visible points, rescaled if necessary.
    slice_input : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    request_id : int
        The identifier of the request from which this was generated.
    """

    indices: npt.NDArray = field(repr=False)
    size: npt.NDArray = field(repr=False)
    slice_input: _SliceInput
    request_id: int


@dataclass(frozen=True)
class _PointSliceRequest:
    """A callable that stores all the input data needed to slice a Points layer.

    This should be treated a deeply immutable structure, even though some
    fields can be modified in place. It is like a function that has captured
    all its inputs already.

    In general, the calling an instance of this may take a long time, so you may
    want to run it off the main thread.

    Attributes
    ----------
    dims : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    data : Any
        The layer's data field, which is the main input to slicing.
    data_slice : _ThickNDSlice
        The slicing coordinates and margins in data space.
    size : array like
        Size of each point. This is used in calculating visibility.
    shown : array like
        Boolean array indicating if each point should be shown.
    others
        See the corresponding attributes in `Layer` and `Points`.
    """

    slice_input: _SliceInput
    data: Any = field(repr=False)
    data_slice: _ThickNDSlice = field(repr=False)
    projection_mode: PointsProjectionMode
    size: npt.NDArray = field(repr=False)
    shown: npt.NDArray = field(repr=False)
    id: int = field(default_factory=_next_request_id)

    def __call__(self) -> _PointSliceResponse:
        # Return early if no data
        if len(self.data) == 0:
            return _PointSliceResponse(
                indices=np.empty(0, dtype=int),
                size=np.empty(0, dtype=float),
                slice_input=self.slice_input,
                request_id=self.id,
            )

        not_disp = list(self.slice_input.not_displayed)
        if not not_disp:
            # If we want to display everything, then use all indices.
            # scale is only impacted by not displayed data, therefore 1
            return _PointSliceResponse(
                indices=np.arange(self.data.shape[0], dtype=int),
                size=self.size,
                slice_input=self.slice_input,
                request_id=self.id,
            )

        indices, size = self._get_slice_data(not_disp)

        return _PointSliceResponse(
            indices=indices,
            size=size,
            slice_input=self.slice_input,
            request_id=self.id,
        )

    def _get_slice_data(
        self, not_disp: list[int]
    ) -> tuple[
        npt.NDArray,
        npt.NDArray,
    ]:
        data_not_disp = self.data[:, not_disp]

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

        inside_slice = np.all(
            (data_not_disp >= low) & (data_not_disp <= high), axis=1
        )

        if self.projection_mode in ('all', 'none'):
            valid_points = inside_slice
            scale = 1
        elif self.projection_mode == 'rescale':
            sizes = self.size[:, np.newaxis] / 2

            # add out of slice points with progressively lower sizes
            dist_from_low = np.abs(data_not_disp - low)
            dist_from_high = np.abs(data_not_disp - high)
            distances = np.minimum(dist_from_low, dist_from_high)
            # anything inside the slice is at distance 0
            distances[inside_slice] = 0

            # display points that "spill" into the slice
            valid_points = np.all(distances <= sizes, axis=1)
            if not np.any(valid_points):
                return (
                    np.empty(0, dtype=int),
                    np.empty(0, dtype=float),
                )

            # rescale size of spilling points based on how much they do
            scale_per_dim = (sizes - distances) / sizes
            scale = np.prod(scale_per_dim, axis=1)

        visible = np.where(valid_points & self.shown)[0].astype(int)
        size = (self.size * scale)[visible]
        return visible, size
