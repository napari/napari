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
        self.face_normals_visual = None
        self.vertex_normals_visual = None
        super().__init__(*args, **kwargs)
        self.attach(self.wireframe_filter)
        self.wireframe_filter.enabled = self.layer.wireframe

    def update_face_normals(self):
        # we have to skirt around the fact that MeshNormals breaks with empty data
        if self.mesh_data._face_normals is not None:
            if self.face_normals_visual is None:
                self.face_normals_visual = MeshNormals(
                    meshdata=self.mesh_data, parent=self
                )
            else:
                self.face_normals_visual.set_data(self.mesh_data)
        elif self.face_normals_visual is not None:
            self.face_normals_visual.parent = None
            self.face_normals_visual = None

    def update_vertex_normals(self):
        # we have to skirt around the fact that MeshNormals breaks with empty data
        if self.mesh_data._vertex_normals is not None:
            if self.vertex_normals_visual is None:
                self.vertex_normals_visual = MeshNormals(
                    meshdata=self.mesh_data, parent=self
                )
            else:
                self.vertex_normals_visual.set_data(self.mesh_data)
        elif self.vertex_normals_visual is not None:
            self.vertex_normals_visual.parent = None
            self.vertex_normals_visual = None
