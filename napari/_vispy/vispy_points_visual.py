from vispy.scene.visuals import Compound, Line, Text

from .markers import Markers


class PointsVisual(Compound):
    def __init__(self):
        self._clipping_planes = None
        super().__init__([Markers(), Markers(), Line(), Text()])

    @property
    def clipping_planes(self):
        return self._clipping_planes

    @clipping_planes.setter
    def clipping_planes(self, value):
        for subv in self._subvisuals:
            if hasattr(subv, 'clipping_planes'):
                subv.clipping_planes = value
        self._clipping_planes = value
