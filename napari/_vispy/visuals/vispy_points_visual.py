from vispy.scene.visuals import Compound, Line, Text

from .markers import Markers
from .vendored.filters.clipping_planes import PlanesClipper


class PointsVisual(Compound):
    """
    Compound vispy visual for point visualization
    """

    def __init__(self):
        self._clip_filter = PlanesClipper()
        super().__init__([Markers(), Markers(), Line(), Text()])

        for subv in self._subvisuals:
            subv.attach(self._clip_filter)

    @property
    def clipping_planes(self):
        return self._clip_filter.clipping_planes

    @clipping_planes.setter
    def clipping_planes(self, value):
        self._clip_filter.clipping_planes = value
