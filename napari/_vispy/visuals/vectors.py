from .clipping_planes_mixin import ClippingPlanesMixin
from .lines import LineQuad


class VectorsVisual(ClippingPlanesMixin, LineQuad):
    """
    Vectors vispy visual with clipping plane functionality
    """
