from vispy.scene.visuals import Compound, Line, Text

from .vendored.filters.clipping_planes import PlanesClipper


class TracksVisual(Compound):
    """
    Compound vispy visual for Track visualization
    """

    def __init__(self):
        self._clip_filter = PlanesClipper()
        super().__init__([Line(), Text(), Line()])

        for subv in self._subvisuals:
            subv.attach(self._clip_filter)

    @property
    def clipping_planes(self):
        return self._clip_filter.clipping_planes

    @clipping_planes.setter
    def clipping_planes(self, value):
        self._clip_filter.clipping_planes = value
