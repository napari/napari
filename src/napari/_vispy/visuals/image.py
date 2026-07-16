from vispy.scene.visuals import Image as BaseImage

from napari._vispy.visuals.util import TextureMixin


# If data is not present, we need bounds to be None (see napari#3517)
class Image(TextureMixin, BaseImage):
    def _compute_bounds(self, axis, view):
        if self._data is None:
            return None
        if axis > 1:
            return (0, 0)

        return (0, self.size[axis])
