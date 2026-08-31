from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from vispy.visuals.filters.clipping_planes import PlanesClipper

if TYPE_CHECKING:
    from vispy.visuals.filters import Filter

    from napari._vispy.utils.qt_font import FontInfo


class _PVisual(Protocol):
    """
    Type for vispy visuals that implement the attach method
    """

    _subvisuals: list[_PVisual] | None
    _clip_filter: PlanesClipper

    def attach(self, filt: Filter, view=None): ...


class ClippingPlanesMixin:
    """
    Mixin class that attaches clipping planes filters to the (sub)visuals
    and provides property getter and setter
    """

    def __init__(self: _PVisual, *args, font_info: FontInfo, **kwargs) -> None:
        clip_filter = PlanesClipper()
        self._clip_filter = clip_filter
        self.font_info = font_info
        super().__init__(*args, **kwargs)

        self.attach(clip_filter)

    @property
    def clipping_planes(self):
        return self._clip_filter.clipping_planes

    @clipping_planes.setter
    def clipping_planes(self, value):
        self._clip_filter.clipping_planes = value
