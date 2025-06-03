from napari._pydantic_compat import Field
from napari.utils.color import ColorValue
from napari.utils.compat import StrEnum
from napari.utils.events import EventedModel


class Orientation(StrEnum):
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'


class OverlayTiling(EventedModel):
    top_left: Orientation = Orientation.VERTICAL
    top_center: Orientation = Orientation.VERTICAL
    top_right: Orientation = Orientation.HORIZONTAL
    bottom_left: Orientation = Orientation.HORIZONTAL
    bottom_center: Orientation = Orientation.VERTICAL
    bottom_right: Orientation = Orientation.VERTICAL


class Canvas(EventedModel):
    background_color: ColorValue | None = None
    overlay_tiling: OverlayTiling = Field(default_factory=OverlayTiling)
