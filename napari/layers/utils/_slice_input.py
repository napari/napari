from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from napari.utils.misc import reorder_after_dim_reduction
from napari.utils.transforms import Affine
from napari.utils.translations import trans


@dataclass(frozen=True)
class _SliceInput:
    """Encapsulates the input needed for slicing a layer.

    An instance of this should be associated with a layer and some of the values
    in ``Viewer.dims`` when slicing a layer.
    """

    # The number of dimensions to be displayed in the slice.
    ndisplay: int
    # The point in layer world coordinates that defines the slicing plane.
    # Only the elements in the non-displayed dimensions have meaningful values.
    point: Tuple[float, ...]
    # The layer dimension indices in the order they are displayed.
    # A permutation of the ``range(self.ndim)``.
    # The last ``self.ndisplay`` dimensions are displayed in the canvas.
    order: Tuple[int, ...]

    @property
    def ndim(self) -> int:
        """The dimensionality of the associated layer."""
        return len(self.order)

    @property
    def displayed(self) -> List[int]:
        """The layer dimension indices displayed in this slice."""
        return list(self.order[-self.ndisplay :])

    @property
    def not_displayed(self) -> List[int]:
        """The layer dimension indices not displayed in this slice."""
        return list(self.order[: -self.ndisplay])

    def with_ndim(self, ndim: int) -> _SliceInput:
        """Returns a new instance with the given number of layer dimensions."""
        old_ndim = self.ndim
        if old_ndim > ndim:
            point = self.point[-ndim:]
            order = reorder_after_dim_reduction(self.order[-ndim:])
        elif old_ndim < ndim:
            point = (0,) * (ndim - old_ndim) + self.point
            order = tuple(range(ndim - old_ndim)) + tuple(
                o + ndim - old_ndim for o in self.order
            )
        else:
            point = self.point
            order = self.order
        return _SliceInput(ndisplay=self.ndisplay, point=point, order=order)

    def data_indices(
        self, world_to_data: Affine, round_index: bool = True
    ) -> Tuple[Union[int, float, slice]]:
        """Transforms this into indices that can be used to slice layer data.

        The elements in non-displayed dimensions will be real numbers.
        The elements in displayed dimensions will be ``slice(None)``.
        """
        if not self.is_orthogonal(world_to_data):
            warnings.warn(
                trans._(
                    'Non-orthogonal slicing is being requested, but is not fully supported. Data is displayed without applying an out-of-slice rotation or shear component.',
                    deferred=True,
                ),
                category=UserWarning,
            )

        slice_world_to_data = world_to_data.set_slice(self.not_displayed)
        world_pts = [self.point[ax] for ax in self.not_displayed]
        data_pts = slice_world_to_data(world_pts)
        if round_index:
            # A round is taken to convert these values to slicing integers
            data_pts = np.round(data_pts).astype(int)

        indices = [slice(None)] * self.ndim
        for i, ax in enumerate(self.not_displayed):
            indices[ax] = data_pts[i]

        return tuple(indices)

    def is_orthogonal(self, world_to_data: Affine) -> bool:
        """Returns True if this slice represents an orthogonal slice through a layer's data, False otherwise."""
        # Subspace spanned by non displayed dimensions
        non_displayed_subspace = np.zeros(self.ndim)
        for d in self.not_displayed:
            non_displayed_subspace[d] = 1
        # Map subspace through inverse transform, ignoring translation
        world_to_data = Affine(
            ndim=self.ndim,
            linear_matrix=world_to_data.linear_matrix,
            translate=None,
        )
        mapped_nd_subspace = world_to_data(non_displayed_subspace)
        # Look at displayed subspace
        displayed_mapped_subspace = (
            mapped_nd_subspace[d] for d in self.displayed
        )
        # Check that displayed subspace is null
        return all(abs(v) < 1e-8 for v in displayed_mapped_subspace)
