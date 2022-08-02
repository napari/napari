from ...utils.events import EventedModel
from ...utils.transforms import Transform
from .._viewer_constants import CanvasPosition


class Overlay(EventedModel):
    visible: bool = False
    opacity: float = 1
    order: int = 1e6

    def __hash__(self):
        return id(self)


class CanvasOverlay(Overlay):
    position: CanvasPosition = CanvasPosition.BOTTOM_RIGHT


class SceneOverlay(Overlay):
    transform: Transform


class LayerOverlay(SceneOverlay):
    pass
