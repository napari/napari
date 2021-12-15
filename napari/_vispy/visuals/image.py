from vispy.scene.visuals import Image as BaseImage


# If data is not present, we need bounds to be None (see napari#3517)
class Image(BaseImage):
    def _compute_bounds(self, axis, view):
        if self._data is None:
            return None
        elif axis > 1:
            return (0, 0)
        else:
            return (0, self.size[axis])
