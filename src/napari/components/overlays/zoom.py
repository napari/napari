"""Zoom-box."""

from __future__ import annotations

import numpy as np

from napari._pydantic_compat import validator
from napari.components.overlays.base import CanvasOverlay
from napari.utils.events import Event


class ZoomOverlay(CanvasOverlay):
    """A box that can be used to select and transform objects.

    Attributes
    ----------
    bounds : 2-tuple of 2-tuples
        Corners at top left and bottom right in layer coordinates.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    canvas_positions: tuple[tuple[float, float], tuple[float, float]] = (
        (0, 0),
        (0, 0),
    )
    data_positions: tuple[tuple[float, ...], tuple[float, ...]] = (
        (0, 0),
        (0, 0),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events.add(zoom=Event)

    @validator(
        'canvas_positions',
        'data_positions',
        pre=True,
        always=True,
        allow_reuse=True,
    )
    def _validate_bounds(
        cls, v: tuple[tuple[float, ...], tuple[float, ...]]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        tup_1, tup_2 = v
        return tuple(tup_1), tuple(tup_2)

    def data_extents(
        self, displayed: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the extents of the overlay in the scene coordinates.

        Parameters
        ----------
        displayed : tuple[int, ...]
            Axes that are currently displayed in the viewer.

        Returns
        -------
        extents : tuple of 4 floats
            The extents of the overlay in the scene coordinates.
            dim1_min, dim1_max, dim2_min, dim2_max
        """
        top_left, bot_right = self.data_positions
        top_left = np.array([top_left[i] for i in displayed])
        bot_right = np.array([bot_right[i] for i in displayed])
        extents = np.vstack((top_left, bot_right))
        mins = np.min(extents, axis=0)
        maxs = np.max(extents, axis=0)
        return mins, maxs

    def canvas_extents(self) -> tuple[float, float, float, float]:
        """Bounds."""
        top_left, bot_right = self.canvas_positions

        dim1_min = min(top_left[0], bot_right[0])
        dim1_max = max(top_left[0], bot_right[0])
        dim2_min = min(top_left[1], bot_right[1])
        dim2_max = max(top_left[1], bot_right[1])
        return dim1_min, dim1_max, dim2_min, dim2_max
