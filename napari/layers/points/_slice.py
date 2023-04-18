from dataclasses import dataclass, field
from typing import Any

import numpy as np

from napari.layers.utils._slice_input import _SliceInput


@dataclass(frozen=True)
class _PointSliceResponse:
    """Contains all the output data of slicing an image layer.

    Attributes
    ----------
    indices : array like
        Indices of the sliced Points data.
    scale: array like or none
        Used to scale the sliced points for visualization.
        Should be broadcastable to indices.
    dims : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    """

    indices: np.ndarray = field(repr=False)
    scale: Any = field(repr=False)
    dims: _SliceInput


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
    dims_indices : tuple of ints or slices
        The slice indices in the layer's data space.
    size : array like
        Size of each point. This is used in calculating visibility.
    others
        See the corresponding attributes in `Layer` and `Image`.
    """

    dims: _SliceInput
    data: Any = field(repr=False)
    dims_indices: Any = field(repr=False)
    size: Any = field(repr=False)
    out_of_slice_display: bool = field(repr=False)

    def __call__(self) -> _PointSliceResponse:
        # Return early if no data
        if len(self.data) == 0:
            return _PointSliceResponse(
                indices=[], scale=np.empty(0), dims=self.dims
            )

        not_disp = list(self.dims.not_displayed)
        if not not_disp:
            # If we want to display everything, then use all indices.
            # scale is only impacted by not displayed data, therefore 1
            return _PointSliceResponse(
                indices=np.arange(len(self.data), dtype=int),
                scale=1,
                dims=self.dims,
            )

        # We want a numpy array so we can use fancy indexing with the non-displayed
        # indices, but as self.dims_indices can (and often/always does) contain slice
        # objects, the array has dtype=object which is then very slow for the
        # arithmetic below. As Points._round_index is always False, we can safely
        # convert to float to get a major performance improvement.
        not_disp_indices = np.array(self.dims_indices)[not_disp].astype(float)

        if self.out_of_slice_display and self.dims.ndim > 2:
            slice_indices, scale = self._get_out_of_display_slice_data(
                not_disp, not_disp_indices
            )
        else:
            slice_indices, scale = self._get_slice_data(
                not_disp, not_disp_indices
            )

        return _PointSliceResponse(
            indices=slice_indices, scale=scale, dims=self.dims
        )

    def _get_out_of_display_slice_data(self, not_disp, not_disp_indices):
        """This method slices in the out-of-display case."""
        distances = abs(self.data[:, not_disp] - not_disp_indices)
        sizes = self.size[:, np.newaxis] / 2
        matches = np.all(distances <= sizes, axis=1)
        if not np.any(matches):
            return np.empty(0, dtype=int), 1
        size_match = sizes[matches]
        size_match[size_match == 0] = 1
        scale_per_dim = (size_match - distances[matches]) / size_match
        scale_per_dim[size_match == 0] = 1
        scale = np.prod(scale_per_dim, axis=1)
        slice_indices = np.where(matches)[0].astype(int)
        return slice_indices, scale

    def _get_slice_data(self, not_disp, not_disp_indices):
        """This method slices in the simpler case."""
        data = self.data[:, not_disp]
        distances = np.abs(data - not_disp_indices)
        matches = np.all(distances <= 0.5, axis=1)
        slice_indices = np.where(matches)[0].astype(int)
        return slice_indices, 1
