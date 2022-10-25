from dataclasses import dataclass, field
from typing import Any

import numpy as np

from napari.layers.utils._slice_input import _SliceInput
from napari.utils.transforms import Affine


@dataclass(frozen=True)
class _PointSliceResponse:
    indices: np.ndarray = field(repr=False)
    scale: Any = field(repr=False)


@dataclass(frozen=True)
class _PointSliceRequest:
    """Represents a single point slice request.
    This should be treated a deeply immutable structure, even though some
    fields can be modified in place.
    In general, the execute method may take a long time to run, so you may
    want to run it once on a worker thread.
    """

    dims: _SliceInput
    data: Any = field(repr=False)
    dims_indices: Any = field(repr=False)
    data_to_world: Affine = field(repr=False)
    size: Any = field(repr=False)
    out_of_slice_display: bool = field(repr=False)

    def execute(self) -> _PointSliceResponse:

        slice_indices, scale = self._get_slice_data()

        return _PointSliceResponse(indices=slice_indices, scale=scale)

    def _get_slice_data(self):
        """This method does the actual slicing work."""
        not_disp = list(self.dims.not_displayed)
        # We want a numpy array so we can use fancy indexing with the non-displayed
        # indices, but as self.dims_indices can (and often/always does) contain slice
        # objects, the array has dtype=object which is then very slow for the
        # arithmetic below. As Points._round_index is always False, we can safely
        # convert to float to get a major performance improvement.
        not_disp_indices = np.array(self.dims_indices)[not_disp].astype(float)
        if len(self.data) > 0:
            if self.out_of_slice_display and self.dims.ndim > 2:
                distances = abs(self.data[:, not_disp] - not_disp_indices)
                sizes = self.size[:, not_disp] / 2
                matches = np.all(distances <= sizes, axis=1)
                size_match = sizes[matches]
                size_match[size_match == 0] = 1
                scale_per_dim = (size_match - distances[matches]) / size_match
                scale_per_dim[size_match == 0] = 1
                scale = np.prod(scale_per_dim, axis=1)
                slice_indices = np.where(matches)[0].astype(int)
                return slice_indices, scale
            else:
                self.data = self.data[:, not_disp]
                distances = np.abs(self.data - not_disp_indices)
                matches = np.all(distances <= 0.5, axis=1)
                slice_indices = np.where(matches)[0].astype(int)
                return slice_indices, 1
        else:
            return [], np.empty(0)
