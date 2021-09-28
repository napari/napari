from vispy.scene.visuals import Line

from .clipping_planes_mixin import ClippingPlanesMixin


class VectorsVisual(ClippingPlanesMixin, Line):
    """
    Vectors vispy visual with clipping plane functionality
    """
