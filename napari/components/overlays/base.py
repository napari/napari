from ...utils.events import EventedModel
from .._viewer_constants import CanvasPosition


class Overlay(EventedModel):
    visible: bool = False
    opacity: float = 1
    order: int = 1e6


class CanvasOverlay(Overlay):
    position: CanvasPosition = CanvasPosition.BOTTOM_RIGHT


class SceneOverlay(Overlay):
    # TODO: should transform live here?
    pass
