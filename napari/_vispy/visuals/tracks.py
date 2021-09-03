from vispy.scene.visuals import Compound, Line, Text

from .clipping_planes_mixin import ClippingPlanesMixin


class TracksVisual(ClippingPlanesMixin, Compound):
    """
    Compound vispy visual for Track visualization
    """

    def __init__(self):
        super().__init__([Line(), Text(), Line()])
