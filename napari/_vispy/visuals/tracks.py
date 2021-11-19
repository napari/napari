from vispy.scene.visuals import Compound, Line, Text

from ..filters.tracks import TracksFilter
from .clipping_planes_mixin import ClippingPlanesMixin


class TracksVisual(ClippingPlanesMixin, Compound):
    """
    Compound vispy visual for Track visualization with
    clipping planes functionality

    Components:
        - Track lines (vispy.LineVisual)
        - Track IDs (vispy.TextVisual)
        - Graph edges (vispy.LineVisual)
    """

    def __init__(self):
        self.tracks_filter = TracksFilter()

        super().__init__([Line(), Text(), Line()])

        self.attach(self.tracks_filter)

        # text label properties
        self._subvisuals[1].color = 'white'
        self._subvisuals[1].font_size = 8
