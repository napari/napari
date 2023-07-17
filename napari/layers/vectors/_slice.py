from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np

from napari.layers.base._slice import _next_request_id
from napari.layers.utils._slice_input import _SliceInput


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
    dims : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    request_id : int
        The identifier of the request from which this was generated.
    """

    indices: np.ndarray = field(repr=False)
    alphas: Union[np.ndarray, float] = field(repr=False)
    dims: _SliceInput
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
    dims : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    data : Any
        The layer's data field, which is the main input to slicing.
    dims_indices : tuple of ints or slices
        The slice indices in the layer's data space.
    others
        See the corresponding attributes in `Layer` and `Vectors`.
    """

    dims: _SliceInput
    data: Any = field(repr=False)
    dims_indices: Any = field(repr=False)
    length: float = field(repr=False)
    out_of_slice_display: bool = field(repr=False)
    id: int = field(default_factory=_next_request_id)

    def __call__(self) -> _VectorSliceResponse:
        # Return early if no data
        if len(self.data) == 0:
            return _VectorSliceResponse(
                indices=np.empty(0, dtype=int),
                alphas=np.empty(0),
                dims=self.dims,
                request_id=self.id,
            )

        not_disp = list(self.dims.not_displayed)
        if not not_disp:
            # If we want to display everything, then use all indices.
            # alpha is only impacted by not displayed data, therefore 1
            return _VectorSliceResponse(
                indices=np.arange(len(self.data), dtype=int),
                alphas=1,
                dims=self.dims,
                request_id=self.id,
            )

        # We want a numpy array so we can use fancy indexing with the non-displayed
        # indices, but as self.dims_indices can (and often/always does) contain slice
        # objects, the array has dtype=object which is then very slow for the
        # arithmetic below. As Vectors._round_index is always False, we can safely
        # convert to float to get a major performance improvement.
        not_disp_indices = np.array(self.dims_indices)[not_disp].astype(float)

        if self.out_of_slice_display and self.dims.ndim > 2:
            slice_indices, alphas = self._get_out_of_display_slice_data(
                not_disp, not_disp_indices
            )
        else:
            slice_indices, alphas = self._get_slice_data(
                not_disp, not_disp_indices
            )

        return _VectorSliceResponse(
            indices=slice_indices,
            alphas=alphas,
            dims=self.dims,
            request_id=self.id,
        )

    def _get_out_of_display_slice_data(self, not_disp, not_disp_indices):
        """This method slices in the out-of-display case."""
        data = self.data[:, 0, not_disp]
        distances = abs(data - not_disp_indices)
        # get the scaled projected vectors
        projected_lengths = abs(self.data[:, 1, not_disp] * self.length)
        # find where the distance to plane is less than the scaled vector
        matches = np.all(distances <= projected_lengths, axis=1)
        alpha_match = projected_lengths[matches]
        alpha_match[alpha_match == 0] = 1
        alpha_per_dim = (alpha_match - distances[matches]) / alpha_match
        alpha_per_dim[alpha_match == 0] = 1
        alpha = np.prod(alpha_per_dim, axis=1).astype(float)
        slice_indices = np.where(matches)[0].astype(int)
        return slice_indices, alpha

    def _get_slice_data(self, not_disp, not_disp_indices):
        """This method slices in the simpler case."""
        data = self.data[:, 0, not_disp]
        distances = np.abs(data - not_disp_indices)
        matches = np.all(distances <= 0.5, axis=1)
        slice_indices = np.where(matches)[0].astype(int)
        return slice_indices, 1
