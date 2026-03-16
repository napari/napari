from __future__ import annotations

from typing import TYPE_CHECKING

from vispy.scene.visuals import Compound, Line

from napari._vispy.filters.tracks import TracksFilter
from napari._vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin
from napari._vispy.visuals.text import Text

if TYPE_CHECKING:
    from napari._vispy.utils.qt_font import FontInfo


class TracksVisual(ClippingPlanesMixin, Compound):
    """
    Compound vispy visual for Track visualization with
    clipping planes functionality

    Components:
        - Track lines (vispy.LineVisual)
        - Track IDs (vispy.TextVisual)
        - Graph edges (vispy.LineVisual)
    """

    def __init__(self, font_info: FontInfo) -> None:
        self.tracks_filter = TracksFilter()
        self.graph_filter = TracksFilter()

        super().__init__(
            [
                Line(antialias=True),
                Text(font_info=font_info),
                Line(antialias=True),
            ],
            font_info=font_info,
        )

        self._subvisuals[0].attach(self.tracks_filter)
        self._subvisuals[2].attach(self.graph_filter)

        # text label properties
        self._subvisuals[1].color = 'white'
        self._subvisuals[1].font_size = 8
