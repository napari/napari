from vispy.scene.visuals import create_visual_node

from .vendored import MeshVisual
from .vendored.filters.clipping_planes import PlanesClipper

BaseMesh = create_visual_node(MeshVisual)


class Mesh(BaseMesh):
    def __init__(self):
        self._clip_filter = PlanesClipper()
        super().__init__()
        self.attach(self._clip_filter)

    @property
    def clipping_planes(self):
        return self._clip_filter.clipping_planes

    @clipping_planes.setter
    def clipping_planes(self, value):
        self._clip_filter.clipping_planes = value
