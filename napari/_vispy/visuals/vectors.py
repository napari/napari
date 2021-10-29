from .clipping_planes_mixin import ClippingPlanesMixin
from .mesh import Mesh


class VectorsVisual(ClippingPlanesMixin, Mesh):
    """
    Vectors vispy visual with clipping plane functionality
    """
