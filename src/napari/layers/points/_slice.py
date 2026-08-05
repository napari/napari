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
    ) -> tuple[npt.NDArray, npt.NDArray]:

        point, m_left, m_right = self.data_slice[not_disp].as_array()

        if self.projection_mode == PointsProjectionMode.NONE:
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

        data_not_disp = self.data[:, not_disp]
        inside_slice = np.all(
            (data_not_disp >= low) & (data_not_disp <= high), axis=1
        )
        visible = np.where(inside_slice & self.shown)[0].astype(int)

        if not visible.size:
            return (
                np.empty(0, dtype=int),
                np.empty(0, dtype=float),
            )

        size = self.size[visible]

        if self.projection_mode in (
            PointsProjectionMode.RESCALE_LINEAR,
            PointsProjectionMode.RESCALE_SPHERICAL,
        ):
            # rescale size of points based on how far they are from the center
            dist_from_slice = np.abs(data_not_disp[visible] - point)
            # use the highest distance (since if you're far in any dimension then
            # it doesn't matter how close you are in any other)
            min_dist_from_slice = np.min(dist_from_slice, axis=1)

            if self.projection_mode == PointsProjectionMode.RESCALE_LINEAR:
                # linear, closest to the old out_of_slice_display
                max_dist = np.max([np.abs(low - point), np.abs(high - point)])
                size = size * min_dist_from_slice / max_dist
            elif self.projection_mode == PointsProjectionMode.RESCALE_LINEAR:
                # This follows a spherical decay, meaning that while a point's poisition
                # may be in the slice, if the sphere centered on it does not intersect
                # the center of the slice, it will be discarded
                cap_radius_square = (size / 2) ** 2 - dist_from_slice**2

                # slice out ASAP to reduce computations
                valid = cap_radius_square > 0
                visible = visible[valid]
                size = np.sqrt(cap_radius_square[valid]) * 2

        return visible, size
