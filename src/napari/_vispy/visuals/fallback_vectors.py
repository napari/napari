from vispy.scene.visuals import Line

from napari._vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin


class Vectors(ClippingPlanesMixin, Line):
    """Fallback line-based vectors visual."""

    def __init__(self, **kwargs):
        super().__init__(connect='segments', **kwargs)

    def set_data(
        self,
        vertices,
        colors,
        vector_style,
    ):
        super().set_data(pos=vertices, color=colors.repeat(2, axis=0))

    @property
    def width(self):
        return Line.width.fget(self)

    @width.setter
    def width(self, value):
        super().set_data(width=value)
