from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from napari.layers.base._slice import _next_request_id
from napari.layers.points._points_constants import (
    ND_PROJECTION_MODES,
    PointsProjectionMode,
)
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice


def _slice_face_distance(
    dist_from_point: npt.NDArray,
    low: npt.NDArray,
    high: npt.NDArray,
    point: npt.NDArray,
) -> npt.NDArray:
    """Per-dimension signed distance to the face of the slice on that side.

    Negative on the low side of the slice point, and zero when that side has no
    margin.
    """
    return np.where(dist_from_point < 0, low - point, high - point)


def _distance_outside_slice(
    dist_from_point: npt.NDArray,
    low: npt.NDArray,
    high: npt.NDArray,
    point: npt.NDArray,
) -> npt.NDArray:
    """Per-dimension distance from the slice: zero inside, positive outside.

    `low`/`high` already include the +-0.5 fallback applied to a thin slice, so
    this is measured from the effective faces of the slice.
    """
    return np.clip(
        np.abs(dist_from_point)
        - np.abs(_slice_face_distance(dist_from_point, low, high, point)),
        0.0,
        None,
    )


def _disc_diameter_from_slice_distance(
    distance: npt.NDArray, radius: npt.NDArray
) -> npt.NDArray:
    """Diameter of the disc a sphere of `radius` makes at `distance`.

    `distance` is the per-dimension distance from the slice, so a sphere whose
    centre is inside the slice (distance 0) keeps its full diameter. The
    per-dimension portions are multiplied into one, as the other rescaling
    modes do.
    """
    in_slice_portion = np.prod(1 - (distance / radius[:, None]), axis=1)
    radius_segment = (1 - in_slice_portion) * radius
    return 2 * np.sqrt(np.maximum(radius**2 - radius_segment**2, 0.0))


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
        radius = self.size / 2

        if self.projection_mode in ND_PROJECTION_MODES:
            # Determine if a point is in the slice by its extent (point size
            # as a sphere) rather than its position. A member of the slice has
            # any of its extent inside it.
            dist_from_center = data_not_disp - point
            dist_from_slice = _distance_outside_slice(
                dist_from_center, low, high, point
            )
            reaches_slice = np.all(
                dist_from_slice < radius[:, None], axis=1
            ) & (radius > 0)
            visible = np.where(reaches_slice & self.shown)[0].astype(int)
        else:
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
            *ND_PROJECTION_MODES,
        ):
            # our rescaling is relative to the center of the slice, in each dimension
            dist_from_point = data_not_disp[visible] - point
            radius = radius[visible]
            if self.projection_mode == PointsProjectionMode.RESCALE_LINEAR:
                # linear rescaling, closest to the old out_of_slice_display implementation

                # margins can be different, so we need to treat low/high distance independently
                slice_end = _slice_face_distance(
                    dist_from_point, low, high, point
                )
                # if a point is on the face of a slice, the distance would
                # be zero and result in a division by zero, so we remove those
                # points completely
                dist_ratio = np.divide(
                    dist_from_point,
                    slice_end,
                    out=np.zeros(dist_from_point.shape, dtype=float),
                    where=slice_end != 0,
                )
                # we multiply the scales from each dimension into a single one
                scale = np.prod(1 - dist_ratio, axis=1)
                size = size * scale
            elif (
                self.projection_mode == PointsProjectionMode.RESCALE_SPHERICAL
            ):
                # This follows a spherical decay, meaning that while a point's poisition
                # may be in the slice, if the sphere centered on it does not intersect
                # the center of the slice, it will be discarded

                # the length of the radius segment cut by the intersection with the
                # slice center (per dimension) is dist_from_point. When bigger than size
                # in any dimension, then there is no intersection!
                radius_segment = np.abs(dist_from_point)

                # discard points whose radius is bigger than the distance to the slice
                # (no intersection between the sphere and the slice center)
                valid = np.all(radius_segment < radius[:, None], axis=1)
                size = _disc_diameter_from_slice_distance(
                    radius_segment[valid], radius[valid]
                )
                visible = visible[valid]
            else:
                # the _ND modes: the extent reaches the slice (see the
                # membership test above), and the size is the part of that
                # extent which is inside it.
                dist_from_slice = _distance_outside_slice(
                    dist_from_point, low, high, point
                )
                if (
                    self.projection_mode
                    == PointsProjectionMode.RESCALE_SPHERICAL_ND
                ):
                    # the disc the slice cuts out of the point's extent: a point
                    # at the centre of the slice shows its full diameter
                    size = _disc_diameter_from_slice_distance(
                        dist_from_slice, radius
                    )
                else:  # PointsProjectionMode.RESCALE_LINEAR_ND
                    # fade linearly over the point's own radius, so a point
                    # inside the slice keeps its full size
                    scale = np.prod(
                        1 - (dist_from_slice / radius[:, None]), axis=1
                    )
                    size = size * scale

        return visible, size
