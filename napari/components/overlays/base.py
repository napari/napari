from ...utils.events import EventedModel
from ...utils.transforms import Transform
from .._viewer_constants import Position


class Overlay(EventedModel):
    visible: bool = False
    opacity: float = 1
    order: int = 1e6


class CanvasOverlay(Overlay):
    position: Position = Position.BOTTOM_RIGHT


class SceneOverlay(Overlay):
    transform: Transform


class LayerOverlay(SceneOverlay):
    pass
