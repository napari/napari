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

    # Create a compound visual with the following four subvisuals:
    # Markers: corresponding to the vertices of the interaction box or the
    # shapes that are used for highlights.
    # Lines: The lines of the interaction box used for highlights.
    # Mesh: The mesh of the outlines for each shape used for highlights.
    # Mesh: The actual meshes of the shape faces and edges

    def __init__(self):
        super().__init__([Mesh(), Mesh(), Line(), Markers(), Text()])
