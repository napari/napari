from vispy.scene.visuals import Compound, Line, Text

from ..filters.tracks_shader import TrackShader
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
        # build and attach the shader to the track
        # TODO: does this need to be 2 separate shaders?
        self.track_shader = TrackShader()
        self.graph_shader = TrackShader()

        super().__init__([Line(), Text(), Line()])

        self._subvisuals[0].attach(self.track_shader)
        self._subvisuals[2].attach(self.graph_shader)

        # text label properties
        self._subvisuals[1].color = 'white'
        self._subvisuals[1].font_size = 8
