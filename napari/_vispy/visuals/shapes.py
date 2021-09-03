from vispy.scene.visuals import Compound, Line, Text

from .clipping_planes_mixin import ClippingPlanesMixin
from .markers import Markers
from .mesh import Mesh


class ShapesVisual(ClippingPlanesMixin, Compound):
    """
    Compound vispy visual for shapes visualization
    """

    def __init__(self):
        super().__init__([Mesh(), Mesh(), Line(), Markers(), Text()])
