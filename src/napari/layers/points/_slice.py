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
        if not self.shown.size:
            # shortcut if no points are shown
            return (
                np.empty(0, dtype=int),
                np.empty(0, dtype=float),
            )

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

        # only operate on points that are shown
        data_not_disp = self.data[self.shown][:, not_disp]
        dist_from_point = data_not_disp - point
        dist_from_low = data_not_disp - low
        dist_from_high = data_not_disp - high
        below_low = dist_from_low <= 0
        above_high = dist_from_high >= 0
        inside_slice = np.all(~below_low & ~above_high, axis=1)

        if not inside_slice.size and self.projection_mode in (
            PointsProjectionMode.RESCALE_SPHERICAL,
            PointsProjectionMode.RESCALE_SPHERICAL_THICK,
        ):
            # nothing is inside the slice and nothing will be recovered by the spherical spills
            return (
                np.empty(0, dtype=int),
                np.empty(0, dtype=float),
            )

        size = self.size[self.shown]

        if self.projection_mode in (
            PointsProjectionMode.NONE,
            PointsProjectionMode.ALL,
        ):
            size = size[inside_slice]
        elif self.projection_mode == PointsProjectionMode.RESCALE_LINEAR:
            dist = dist_from_point[inside_slice]
            # margins can be different, so we need to treat low/high distance independently
            slice_end = np.where(dist < 0, low - point, high - point)
            # we multiply the scales from each dimension into a single one
            scale = np.prod(1 - (dist / slice_end), axis=1)
            size = size[inside_slice] * scale
        elif self.projection_mode in (
            PointsProjectionMode.RESCALE_SPHERICAL,
            PointsProjectionMode.RESCALE_SPHERICAL_THICK,
        ):
            # we include points whose spherical extent "spills" into the slice
            radius = size / 2

            if self.projection_mode == PointsProjectionMode.RESCALE_SPHERICAL:
                radius_segment = np.abs(dist_from_point)
            elif (
                self.projection_mode
                == PointsProjectionMode.RESCALE_SPHERICAL_THICK
            ):
                radius_segment = np.where(
                    below_low,
                    np.abs(dist_from_low),
                    np.where(above_high, np.abs(dist_from_high), 0),
                )

            # only work on points intersecting the point or margins to calculate the rescaled size
            inside_slice = np.all(radius_segment < radius[:, None], axis=1)
            radius_segment = radius_segment[inside_slice]
            radius = radius[inside_slice]

            # reduce to one dimension by getting the ndimensional "in slice portion"
            # of the point, and multiplying them together
            out_of_slice_portion = np.prod(
                1 - (radius_segment / radius[:, None]), axis=1
            )
            radius_segment = (1 - out_of_slice_portion) * radius

            # radius of the "disc"
            disc_radius = np.sqrt(radius**2 - radius_segment**2)
            size = disc_radius * 2
        else:
            raise NotImplementedError(
                f'projection mode {self.projection_mode} is not implemented'
            )

        idx_visible = np.where(inside_slice)[0].astype(int)

        return idx_visible, size
