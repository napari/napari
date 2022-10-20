"""PointsSliceData class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from ..base import Layer

if TYPE_CHECKING:
    from ...types import ArrayLike


class PointsSliceData:
    """The contents of an PointsSlice.

    Parameters
    ----------
    layer : Layer
        The layer that contains the data.
    indices : Tuple[Optional[slice], ...]
        The indices of this slice.
    data : ndarray
        The data to display in the slice.
    thumbnail_source : ArrayList
        The source used to create the thumbnail for the slice.
    """

    def __init__(
        self,
        layer: Layer,
        indices: Tuple[Optional[slice], ...],
        data: np.ndarray,
        thumbnail_source: ArrayLike,
    ):
        self.layer = layer
        self.indices = indices
        self.data = data
        self.thumbnail_source = thumbnail_source

    def load_sync(self) -> None:
        """Call asarray on our images to load them."""
        self.image = np.asarray(self.image)

        if self.thumbnail_source is not None:
            self.thumbnail_source = np.asarray(self.thumbnail_source)

    def transpose(self, order: tuple) -> None:
        """Transpose our images.

        Parameters
        ----------
        order : tuple
            Transpose the image into this order.
        """
        self.data = np.transpose(self.data, order)

        if self.thumbnail_source is not None:
            self.thumbnail_source = np.transpose(self.thumbnail_source, order)
