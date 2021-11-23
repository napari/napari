from vispy.scene.visuals import Mesh, MeshNormals
from vispy.visuals.filters import WireframeFilter

from .clipping_planes_mixin import ClippingPlanesMixin


class SurfaceVisual(ClippingPlanesMixin, Mesh):
    """
    Surface vispy visual with added:
        - clipping plane functionality
        - wireframe visualisation
        - normals visualisation
    """

    def __init__(self, *args, **kwargs):
        self.wireframe_filter = WireframeFilter()
        self.normals_visual = None
        super().__init__(*args, **kwargs)
        self.attach(self.wireframe_filter)

    def update_normals(self):
        if self.mesh_data is None:
            if self.normals_visual is not None:
                self.normals_visual.parent = None
            self.normals_visual = None
        else:
            self.normals_visual = MeshNormals(
                meshdata=self.mesh_data, parent=self
            )
