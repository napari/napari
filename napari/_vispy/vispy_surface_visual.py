from .mesh import Mesh
from .vendored.filters.clipping_planes import PlanesClipper


class SurfaceVisual(Mesh):
    """
    Compound vispy visual with markers for
    """

    def __init__(self):
        self._clip_filter = PlanesClipper()
        super().__init__()

    @property
    def clipping_planes(self):
        return self._clip_filter.clipping_planes

    @clipping_planes.setter
    def clipping_planes(self, value):
        self._clip_filter.clipping_planes = value
