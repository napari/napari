from vispy.scene.visuals import Mesh, MeshNormals
from vispy.visuals.filters import WireframeFilter

from napari._vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin


class SurfaceVisual(ClippingPlanesMixin, Mesh):
    """
    Surface vispy visual with added:
        - clipping plane functionality
        - wireframe visualisation
        - normals visualisation
    """

    def __init__(self, *args, **kwargs) -> None:
        self.wireframe_filter = WireframeFilter()
        self.face_normals = MeshNormals(primitive='face')
        self.vertex_normals = MeshNormals(primitive='vertex')
        super().__init__(*args, **kwargs)
        self.attach(self.wireframe_filter)
        self.face_normals.parent = self
        self.vertex_normals.parent = self
