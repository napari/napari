from vispy.scene.visuals import create_visual_node

from ..vendored import MarkersVisual

BaseMarkers = create_visual_node(MarkersVisual)


class Markers(BaseMarkers):
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
