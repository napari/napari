import warnings

from napari._vispy.utils.gl import get_gl_extensions
from napari._vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin

GLPLUS = 'geometry_shader' in get_gl_extensions()
# GLPLUS = False

if GLPLUS:
    from napari._vispy.visuals.lines import Line
else:
    from vispy.scene.visuals import Line as OldLine

    warnings.warn(
        'Could not use GL+ backend; some Line functionality may be limited',
        ImportWarning,
    )

    class Line(OldLine):
        def __init__(self, **kwargs):
            self._scaling = False
            super().__init__(
                **kwargs, connect='segments', method='gl', antialias=False
            )

        @property
        def width(self):
            return self._width

        @width.setter
        def width(self, value):
            self.set_data(width=value)
            self._width = value

        @property
        def scaling(self):
            return self._scaling

        @scaling.setter
        def scaling(self, value):
            warnings.warn(
                'Line width cannot be scaled with zoom without GL+ backend',
                RuntimeWarning,
            )


class VectorsVisual(ClippingPlanesMixin, Line):
    """
    Vectors vispy visual with clipping plane functionality
    """
