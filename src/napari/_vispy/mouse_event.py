from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from vispy.app.canvas import MouseEvent

if TYPE_CHECKING:
    import numpy as np

    VectorF64: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]


class NapariMouseEvent(MouseEvent):
    """NapariMouseEvent extends MouseEvent to include additional information.

    Parameters
    ----------
    event : MouseEvent
        The MouseEvent object to extend.
    view_direction : npt.NDArray[np.float64], optional
        The direction of the camera view, or None if not viewing in 3D.
    up_direction : npt.NDArray[np.float64], optional
        The direction of the camera up vector.
    camera_zoom : float
        The camera zoom level.
    position : npt.NDArray[np.float64]
        The position of the mouse in world coordinates.
    dims_displayed : list[int]
        The dimensions displayed in the viewer.
    dims_point : list[float]
        The point in data coordinates that the mouse is over.
    viewbox : tuple[int, int]
        The coordinates of the grid viewbox containing the mouse.
    """

    def __init__(
        self,
        event: MouseEvent,
        view_direction: VectorF64 | None,
        up_direction: VectorF64 | None,
        camera_zoom: float,
        position: VectorF64,
        dims_displayed: list[int],
        dims_point: list[float],
        viewbox: tuple[int, int],
    ) -> None:
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
        self.viewbox = viewbox
