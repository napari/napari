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
                Markers(scaling='visual'),
                Markers(scaling='visual'),
                Line(),
                Text(),
            ]
        )
        self.scaling = True

    @property
    def scaling(self):
        """
        Scaling property for both the markers visuals. If set to true,
        the points rescale based on zoom (i.e: constant world-space size)
        """
        return self._subvisuals[0].scaling == 'visual'

    @scaling.setter
    def scaling(self, value):
        for marker in self._subvisuals[:2]:
            marker.scaling = 'visual' if value else 'fixed'

    @property
    def antialias(self):
        return self._subvisuals[0].antialias

    @antialias.setter
    def antialias(self, value):
        for marker in self._subvisuals[:2]:
            marker.antialias = value

    @property
    def spherical(self):
        return self._subvisuals[0].spherical

    @spherical.setter
    def spherical(self, value):
        self._subvisuals[0].spherical = value

    @property
    def canvas_size_limits(self):
        return self._subvisuals[0].canvas_size_limits

    @canvas_size_limits.setter
    def canvas_size_limits(self, value):
        self._subvisuals[0].canvas_size_limits = value
