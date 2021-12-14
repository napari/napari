from vispy.scene.visuals import Compound, Line, Markers, Text

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
        super().__init__([Markers(), Markers(), Line(), Text()])

    @property
    def symbol(self):
        return self._subvisuals[0].symbol

    @symbol.setter
    def symbol(self, value):
        for subv in self._subvisuals[:2]:
            subv.symbol = value

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
