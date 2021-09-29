from vispy.scene.visuals import Compound, Line, Text

from .clipping_planes_mixin import ClippingPlanesMixin
from .markers import Markers


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

    def __init__(self):
        super().__init__([Markers(), Markers(), Line(), Text()])

    @property
    def symbol(self):
        return self._subvisuals[0]._symbol

    @symbol.setter
    def symbol(self, value):
        for marker in self._subvisuals[:2]:
            marker.symbol = value

    @property
    def scaling(self):
        return self._subvisuals[0]._scaling

    @scaling.setter
    def scaling(self, value):
        for marker in self._subvisuals[:2]:
            marker.scaling = value

    @property
    def antialias(self):
        return self._subvisuals[0]._antialias

    @antialias.setter
    def antialias(self, value):
        for marker in self._subvisuals[:2]:
            marker.antialias = value

    @property
    def spherical(self):
        return self._subvisuals[0]._spherical

    @spherical.setter
    def spherical(self, value):
        self._subvisuals[0].spherical = value
