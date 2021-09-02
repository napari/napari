from vispy.scene.visuals import create_visual_node

from .vendored import MarkersVisual
from .vendored.filters.clipping_planes import PlanesClipper

BaseMarkers = create_visual_node(MarkersVisual)


class Markers(BaseMarkers):
    def __init__(self):
        self._clip_filter = PlanesClipper()
        super().__init__()
        self.attach(self._clip_filter)

    @property
    def clipping_planes(self):
        return self._clip_filter.clipping_planes

    @clipping_planes.setter
    def clipping_planes(self, value):
        self._clip_filter.clipping_planes = value

    # needed for entering 3D rendering mode when a points
    # layer is invisible and the self._data property is None
    def _compute_bounds(self, axis, view):
        if self._data is None:
            return None
        pos = self._data['a_position']
        if pos is None:
            return None
        if pos.shape[1] > axis:
            return (pos[:, axis].min(), pos[:, axis].max())
        else:
            return (0, 0)
