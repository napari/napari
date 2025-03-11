from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vispy.app.canvas import MouseEvent

if TYPE_CHECKING:
    import numpy.typing as npt


class NapariMouseEvent(MouseEvent):
    """NapariMouseEvent extends MouseEvent to include additional information.

    Parameters
    ----------
    event : MouseEvent
        The MouseEvent object to extend.
    view_direction : npt.NDArray[np.float64]
        The direction of the camera view.
    up_direction : Optional[np.ndarray]
        The direction of the camera up vector.
    camera_zoom : float
        The camera zoom level.
    position : tuple[float, float]
        The position of the mouse in the canvas mapped to data coordinates.
    dims_displayed : list[int]
        The dimensions displayed in the viewer.
    dims_point : list[float]
        The point in data coordinates that the mouse is over.
    """

    def __init__(
        self,
        event: MouseEvent,
        view_direction: npt.NDArray[np.float64],
        up_direction: np.ndarray | None,
        camera_zoom: float,
        position: tuple[float, float],
        dims_displayed: list[int],
        dims_point: list[float],
    ):
        public_attrs = {
            k: v for k, v in event.__dict__.items() if not k.startswith('_')
        }
        super().__init__(
            type=event.type,
            pos=event.pos,
            button=event.button,
            buttons=event.buttons,
            modifiers=event.modifiers,
            delta=event.delta,
            last_event=event.last_event,
            press_event=event.press_event,
            native=event.native,
            **public_attrs,
        )
        self.view_direction = view_direction
        self.up_direction = up_direction
        self.camera_zoom = camera_zoom
        self.position = position
        self.dims_displayed = dims_displayed
        self.dims_point = dims_point
