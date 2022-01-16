from vispy.scene.visuals import Compound, Line, Markers, Text

from ..filters.points_clamp_size import ClampSizeFilter
from .clipping_planes_mixin import ClippingPlanesMixin


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
        self.clamp_filter = ClampSizeFilter()
        super().__init__([Markers(), Markers(), Line(), Text()])
        self.attach(self.clamp_filter)
        self.scaling = True

    @property
    def symbol(self):
        return self._subvisuals[0].symbol

    @symbol.setter
    def symbol(self, value):
        for marker in self._subvisuals[:2]:
            marker.symbol = value

    @property
    def scaling(self):
        """
        Scaling property for both the markers visuals. If set to true,
        the points rescale based on zoom (i.e: constant world-space size)
        """
        return self._subvisuals[0].scaling

    @scaling.setter
    def scaling(self, value):
        for marker in self._subvisuals[:2]:
            marker.scaling = value

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
