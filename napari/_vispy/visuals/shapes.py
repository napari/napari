from vispy.scene.visuals import Compound, Line, Text

from .clipping_planes_mixin import ClippingPlanesMixin
from .markers import Markers
from .mesh import Mesh


class ShapesVisual(ClippingPlanesMixin, Compound):
    """
    Compound vispy visual for shapes visualization with
    clipping planes functionality

    Components:
        - Mesh for shape faces (vispy.MeshVisual)
        - Mesh for highlights (vispy.MeshVisual)
        - Lines for highlights (vispy.LineVisual)
        - Vertices for highlights (vispy.MarkersVisual)
        - Text labels (vispy.TextVisual)
    """

    def __init__(self):
        super().__init__([Mesh(), Mesh(), Line(), Markers(), Text()])
