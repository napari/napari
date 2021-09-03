from vispy.scene.visuals import Compound, Line, Text

from .markers import Markers
from .mesh import Mesh
from .vendored.filters.clipping_planes import PlanesClipper


class ShapesVisual(Compound):
    """
    Compound vispy visual for shapes visualization
    """

    def __init__(self):
        self._clip_filter = PlanesClipper()
        super().__init__([Mesh(), Mesh(), Line(), Markers(), Text()])

        for subv in self._subvisuals:
            subv.attach(self._clip_filter)

    @property
    def clipping_planes(self):
        return self._clip_filter.clipping_planes

    @clipping_planes.setter
    def clipping_planes(self, value):
        self._clip_filter.clipping_planes = value
