from vispy.scene.visuals import create_visual_node

from ..vendored import MeshVisual
from .clipping_planes_mixin import ClippingPlanesMixin

BaseMesh = create_visual_node(MeshVisual)


class Mesh(ClippingPlanesMixin, BaseMesh):
    pass
