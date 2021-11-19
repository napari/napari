from .clipping_planes_mixin import ClippingPlanesMixin
from .mesh import Mesh


class SurfaceVisual(ClippingPlanesMixin, Mesh):
    """
    Surface vispy visual with clipping plane functionality
    """
