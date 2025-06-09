"""Zoom-box."""

import typing as ty

from napari.components.overlays.base import SceneOverlay
from napari.layers.utils.interaction_box import InteractionBoxHandle


class ZoomOverlay(SceneOverlay):
    """A box that can be used to select and transform objects.

    Attributes
    ----------
    bounds : 2-tuple of 3-tuples
        Corners at top left and bottom right in layer coordinates.
    handles : bool
        Whether to show the handles for transfomation or just the box.
    selected_handle : Optional[InteractionBoxHandle]
        The currently selected handle.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0, 0, 0),
        (0, 0, 0),
    )
    handles: bool = False
    selected_handle: ty.Optional[InteractionBoxHandle] = None

    def extents(
        self, displayed: tuple[int, ...]
    ) -> tuple[float, float, float, float]:
        """Return the extents of the overlay in the scene coordinates.

        Returns
        -------
        extents : tuple of 4 floats
            The extents of the overlay in the scene coordinates.
            x_min, x_max, y_min, y_max
        """
        top_left, bot_right = self.bounds
        top_left = tuple([top_left[i] for i in displayed])
        bot_right = tuple([bot_right[i] for i in displayed])

        dim2_min = min(top_left[0], bot_right[0])
        dim2_max = max(top_left[0], bot_right[0])
        dim1_min = min(top_left[1], bot_right[1])
        dim1_max = max(top_left[1], bot_right[1])
        return dim1_min, dim1_max, dim2_min, dim2_max
