from typing import Optional, Tuple

from napari.components.overlays._interaction_box_constants import (
    InteractionBoxHandle,
)
from napari.components.overlays.base import CanvasOverlay
from napari.utils.geometry import bounding_box_from_contained_points


class InteractionBoxOverlay(CanvasOverlay):
    """A box that can be used to select or transform layers or objects.

    Attributes
    ----------
    visible : bool
        If the box is are visible or not.
    bounds : 2-tuple of 2-tuples
        Corners at bottom left and top right.
    handles : bool
        Whether to show the handles for transfomation or just the box.
    selected_handle : Optional[InteractionBoxHandle]
        The currently selected handle.
    """

    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 0), (0, 0))
    handles: bool = False
    selected_vertex: Optional[InteractionBoxHandle] = None

    def update_from_points(self, points):
        """Create as a bounding box of the given points"""
        self.bounds = bounding_box_from_contained_points(points)
