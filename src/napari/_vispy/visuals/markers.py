import logging

from vispy import use
from vispy.scene.visuals import Markers as BaseMarkers

logger = logging.getLogger(__name__)

try:
    use(gl='gl+')
except RuntimeError:
    logger.warning(
        'Could not use gl+ for instanced rendering of points.'
        'Falling back to GL point rendering.'
    )

    rendering_method = 'points'
else:
    rendering_method = 'instanced'


class Markers(BaseMarkers):
    def __init__(self, *args, method=rendering_method, **kwargs) -> None:
        super().__init__(*args, method=method, **kwargs)

    def _compute_bounds(self, axis, view):
        # needed for entering 3D rendering mode when a points
        # layer is invisible and the self._data property is None
        if self._data is None:
            return None
        pos = self._data['a_position']
        if pos is None or pos.size == 0:
            return None
        if pos.shape[1] > axis:
            return (pos[:, axis].min(), pos[:, axis].max())

        return (0, 0)
