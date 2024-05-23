from __future__ import annotations

from vispy.scene.visuals import Compound, Line, Text

from napari._vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin
from napari._vispy.visuals.markers import Markers


class PointsVisual(ClippingPlanesMixin, Compound):
    """
    Compound vispy visual for point visualization with
    clipping planes functionality

    Components:
        - Markers for points (vispy.MarkersVisual)
        - Markers for selection highlights (vispy.MarkersVisual)
        - Lines for highlights (vispy.LineVisual)
        - Text labels (vispy.TextVisual)
    """

    def __init__(self) -> None:
        super().__init__(
            [
                Markers(),
                Markers(),
                Line(),
                Text(),
            ]
        )
        self.scaling = True

    @property
    def points_markers(self) -> Markers:
        """Points markers visual"""
        return self._subvisuals[0]

    @property
    def selection_markers(self) -> Markers:
        """Highlight markers visual"""
        return self._subvisuals[1]

    @property
    def highlight_lines(self) -> Line:
        """Highlight lines visual"""
        return self._subvisuals[2]

    @property
    def text(self) -> Text:
        """Text labels visual"""
        return self._subvisuals[3]

    @property
    def scaling(self) -> bool:
        """
        Scaling property for both the markers visuals. If set to true,
        the points rescale based on zoom (i.e: constant world-space size)
        """
        return self.points_markers.scaling == 'visual'

    @scaling.setter
    def scaling(self, value: bool) -> None:
        scaling_txt = 'visual' if value else 'fixed'
        self.points_markers.scaling = scaling_txt
        self.selection_markers.scaling = scaling_txt

    @property
    def antialias(self) -> float:
        return self.points_markers.antialias

    @antialias.setter
    def antialias(self, value: float) -> None:
        self.points_markers.antialias = value
        self.selection_markers.antialias = value

    @property
    def spherical(self) -> bool:
        return self.points_markers.spherical

    @spherical.setter
    def spherical(self, value: bool) -> None:
        self.points_markers.spherical = value

    @property
    def canvas_size_limits(self) -> tuple[int, int]:
        return self.points_markers.canvas_size_limits

    @canvas_size_limits.setter
    def canvas_size_limits(self, value: tuple[int, int]) -> None:
        self.points_markers.canvas_size_limits = value
        self.selection_markers.canvas_size_limits = value
