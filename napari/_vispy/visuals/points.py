from vispy.scene.visuals import Compound, Line, Text

from .clipping_planes_mixin import ClippingPlanesMixin
from .markers import Markers


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

    # Create a compound visual with the following four subvisuals:
    # Lines: The lines of the interaction box used for highlights.
    # Markers: The the outlines for each point used for highlights.
    # Markers: The actual markers of each point.

    def __init__(self):
        super().__init__([Markers(), Markers(), Line(), Text()])
