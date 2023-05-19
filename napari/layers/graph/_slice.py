from dataclasses import dataclass, field
from typing import Any

import numpy as np
from napari_graph import BaseGraph

from napari.layers.points._slice import _PointSliceRequest
from napari.layers.utils._slice_input import _SliceInput


@dataclass(frozen=True)
class _GraphSliceResponse:
    """Contains all the output data of slicing an graph layer.

    Attributes
    ----------
    indices : array like
        Indices of the sliced *nodes* data.
    edge_indices : array like
        Indices of the slice nodes for each *edge*.
    scale: array like or none
        Used to scale the sliced points for visualization.
        Should be broadcastable to indices.
    dims : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    """

    indices: np.ndarray = field(repr=False)
    edges_indices: np.ndarray = field(repr=False)
    scale: Any = field(repr=False)
    dims: _SliceInput


class _GraphSliceRequest(_PointSliceRequest):
    data: BaseGraph = field(repr=False)  # updating typing

    @property
    def _points_data(self) -> np.ndarray:
        return self.data._coords

    def _edge_indices(self, node_indices: np.ndarray) -> np.ndarray:
        """
        Node indices of pair nodes for each valid edge.
        An edge is valid when both nodes are present.

        NOTE:
        this could be computed in a single shot by rewriting
        _get_out_of_display_slice_data
        _get_slice_data
        """
        mask = np.zeros(len(self.data), dtype=bool)
        mask[node_indices] = True
        _, edges = self.data.get_edges_buffers(is_buffer_domain=True)
        edges_view = edges[
            np.logical_and(mask[edges[:, 0]], mask[edges[:, 1]])
        ]
        return edges_view

    def __call__(self) -> _GraphSliceResponse:
        # Return early if no data
        if len(self.data) == 0:
            return _GraphSliceResponse(
                indices=[],
                edges_indices=[],
                scale=np.empty(0),
                dims=self.dims,
            )

        not_disp = list(self.dims.not_displayed)
        if not not_disp:
            # If we want to display everything, then use all indices.
            # scale is only impacted by not displayed data, therefore 1
            node_indices = np.arange(len(self.data))
            return _GraphSliceResponse(
                indices=node_indices,
                edges_indices=self._edge_indices(node_indices),
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

        return _GraphSliceResponse(
            indices=slice_indices,
            edges_indices=self._edge_indices(slice_indices),
            scale=scale,
            dims=self.dims,
        )
