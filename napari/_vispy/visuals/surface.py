from vispy.scene.visuals import Mesh

from .clipping_planes_mixin import ClippingPlanesMixin


class SurfaceVisual(ClippingPlanesMixin, Mesh):
    """
    Surface vispy visual with clipping plane functionality
    """
