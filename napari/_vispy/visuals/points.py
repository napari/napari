from vispy.scene.visuals import Compound, Line, Text

from .clipping_planes_mixin import ClippingPlanesMixin
from .markers import Markers


class PointsVisual(ClippingPlanesMixin, Compound):
    """
    Compound vispy visual for point visualization
    """

    def __init__(self):
        super().__init__([Markers(), Markers(), Line(), Text()])
