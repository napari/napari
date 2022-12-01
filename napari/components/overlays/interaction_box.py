from typing import Optional, Tuple

from napari.components.overlays._interaction_box_constants import (
    InteractionBoxHandle,
)
from napari.components.overlays.base import SceneOverlay
from napari.utils.geometry import bounding_box_from_contained_points


class SelectionBoxOverlay(SceneOverlay):
    """A box that can be used to select and transform objects.

    Attributes
    ----------
    visible : bool
        If the box is are visible or not.
    bounds : 2-tuple of 2-tuples
        Corners at top left and bottom right in layer coordinates.
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


class TransformBoxOverlay(SceneOverlay):
    """A box that can be used to transform layers.

    Attributes
    ----------
    visible : bool
        If the box is are visible or not.
    """

    selected_vertex: Optional[InteractionBoxHandle] = None
