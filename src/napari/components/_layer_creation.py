"""Spatial metadata for new annotation layers."""

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from napari.layers import Layer


def new_layer_metadata(
    source: 'Layer', ndim: int | None = None
) -> dict[str, Any]:
    """Copy transforms, rejecting reductions that couple discarded axes."""
    physical = source._transforms['data2physical']
    affine = source.affine.affine_matrix
    if ndim is not None and ndim != source.ndim:
        if not 2 <= ndim <= source.ndim:
            raise ValueError(
                'New annotation dimensionality must be between 2 and the reference dimensionality.'
            )
        start = source.ndim - ndim
        for matrix in (physical.linear_matrix, source.affine.linear_matrix):
            if np.any(matrix[:start, start:]) or np.any(
                matrix[start:, :start]
            ):
                raise ValueError('Cannot discard axes coupled by a transform.')
        axes = list(range(start, source.ndim))
        physical = physical.set_slice(axes)
        # Affine.set_slice treats small off-diagonal entries as zero.
        homogeneous_axes = [*axes, source.ndim]
        affine = affine[np.ix_(homogeneous_axes, homogeneous_axes)]
    return {
        'scale': physical.scale,
        'translate': physical.translate,
        'rotate': physical.rotate,
        'shear': physical.shear,
        'units': physical.units,
        'axis_labels': physical.axis_labels,
        'affine': affine,
    }
