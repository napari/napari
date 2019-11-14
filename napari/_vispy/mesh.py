from vispy.scene.visuals import Mesh as BaseMesh


class Mesh(BaseMesh):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_bounds(self, axis, view):
        if self._bounds is None:
            return None
        if axis >= len(self._bounds):
            return (0, 0)
        else:
            return self._bounds[axis]
