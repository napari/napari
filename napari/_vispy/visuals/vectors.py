# import warnings

from ..utils.gl import get_gl_extensions
from .clipping_planes_mixin import ClippingPlanesMixin

GLPLUS = 'geometry_shader' in get_gl_extensions()
# GLPLUS = False  # TODO: remove this

if GLPLUS:
    from .lines import Line
else:
    from vispy.scene.visuals import Line as OldLine

    # TODO uncomment this before merge, it's just to test ci works fine
    # warnings.warn(
    # 'Could not use GL+ backend; some Line functionality may be limited',
    # ImportWarning,
    # )

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
            # TODO uncomment this before merge, it's just to test ci works fine
            # warnings.warn(
            # 'Line width cannot be scaled with zoom without GL+ backend',
            # RuntimeWarning,
            # )
            pass


class VectorsVisual(ClippingPlanesMixin, Line):
    """
    Vectors vispy visual with clipping plane functionality
    """
