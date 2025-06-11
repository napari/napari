"""Zoom-box."""

from __future__ import annotations

from napari._pydantic_compat import validator
from napari.components.overlays.base import SceneOverlay
from napari.layers.utils.interaction_box import InteractionBoxHandle
from napari.utils.events import Event
from napari.utils.misc import ensure_n_tuple


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
    selected_handle: InteractionBoxHandle | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events.add(zoom=Event)

    @validator('bounds', pre=True, always=True, allow_reuse=True)
    def _validate_bounds(
        cls, v: tuple[tuple[float, ...], tuple[float, ...]]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        tup_1, tup_2 = v
        return ensure_n_tuple(tup_1, n=3, before=False), ensure_n_tuple(
            tup_2, n=3, before=False
        )

    def extents(
        self, displayed: tuple[int, ...]
    ) -> tuple[float, float, float, float, float, float]:
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
        top_left, bot_right = self.bounds
        top_left = tuple([top_left[i] for i in displayed])
        bot_right = tuple([bot_right[i] for i in displayed])

        dim1_min = min(top_left[0], bot_right[0])
        dim1_max = max(top_left[0], bot_right[0])
        dim2_min = min(top_left[1], bot_right[1])
        dim2_max = max(top_left[1], bot_right[1])
        dim3_min, dim3_max = 1, 1
        if len(displayed) == 3:
            dim3_min = min(top_left[2], bot_right[2])
            dim3_max = max(top_left[2], bot_right[2])
        return dim1_min, dim1_max, dim2_min, dim2_max, dim3_min, dim3_max
