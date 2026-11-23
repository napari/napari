from dataclasses import dataclass, field
from threading import Event
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from napari.layers._scalar_field._slice import (
    _ScalarFieldSliceRequest,
    _ScalarFieldSliceResponse,
)
from napari.layers.base._base_constants import BaseProjectionMode
from napari.types import ArrayLike
from napari.utils.misc import _is_array_type

if TYPE_CHECKING:
    from weakref import ReferenceType

_MAX_PAINT_REGIONS = 32
_MAX_PAINT_BYTES = 64 * 1024


@dataclass(frozen=True)
class _LabelsSliceRequest(_ScalarFieldSliceRequest):
    read_started: Event = field(
        default_factory=Event, compare=False, repr=False
    )

    def __call__(self) -> _ScalarFieldSliceResponse:
        self.read_started.set()
        return super().__call__()

    @staticmethod
    def _project_slice(
        data: ArrayLike, axis: tuple[int, ...], mode: BaseProjectionMode
    ) -> npt.NDArray:
        """Project a thick slice along axis based on mode."""
        raise NotImplementedError


class _PendingLabelsSlice:
    """Collect edits on the main thread until this request is delivered."""

    def __init__(
        self, request: _LabelsSliceRequest, source: 'ReferenceType[object]'
    ) -> None:
        self.request_id = request.id
        self.read_started = request.read_started
        self.source = source
        self._plane: int | None = None
        self.regions: list[tuple[tuple[slice, ...], np.ndarray]] = []
        self._nbytes = 0

        # A single-scale ZYX Zarr slice is a private YX array. Use the
        # request's data coordinates, which already account for transforms.
        if (
            not request.multiscale
            and request.projection_mode == 'none'
            and request.slice_input.order == (0, 1, 2)
            and request.slice_input.ndisplay == 2
            and _is_array_type(request.data, 'zarr.Array')
        ):
            plane = request._indices(request.data_slice)[0]
            if isinstance(plane, int):
                self._plane = plane

    def record_region(
        self, key: tuple[slice, ...] | None, region: np.ndarray | None
    ) -> bool:
        """Return whether the incoming plane can still be repaired in memory."""
        if key is None or region is None or self._plane is None:
            return False
        if not key[0].start <= self._plane < key[0].stop:
            return True  # The edit was on another plane.

        pixels = region[self._plane - key[0].start]
        if (
            len(self.regions) >= _MAX_PAINT_REGIONS
            or self._nbytes + pixels.nbytes > _MAX_PAINT_BYTES
        ):
            return False
        self.regions.append((key[1:], pixels.copy()))
        self._nbytes += pixels.nbytes
        return True
