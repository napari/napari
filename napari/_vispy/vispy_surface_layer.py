from vispy.scene.visuals import Mesh as MeshNode
from .vispy_base_layer import VispyBaseLayer


class VispySurfaceLayer(VispyBaseLayer):
    def __init__(self, layer):
        node = MeshNode()
        super().__init__(layer, node)

        self.layer.events.colormap.connect(
            lambda e: self._on_colormap_change()
        )
        self.layer.events.contrast_limits.connect(
            lambda e: self._on_contrast_limits_change()
        )

        self.reset()

    def _on_data_change(self):
        if len(self.layer._data_view) == 0 or len(self.layer._view_faces) == 0:
            vertices = np.zeros(3, 2)
            faces = [0, 1, 2]
        else:
            vertices = self.layer._data_view[:, ::-1] + 0.5
            faces = self.layer._view_faces

        self.node.set_data(
            vertices=vertices,
            faces=faces,
            vertex_values=self.layer._view_values,
        )
        self.node.update()

    def _on_colormap_change(self):
        cmap = self.layer.colormap[1]
        if self.layer.dims.ndisplay == 3:
            self.node.view_program['texture2D_LUT'] = (
                cmap.texture_lut() if (hasattr(cmap, 'texture_lut')) else None
            )
        self.node.cmap = cmap

    def _on_contrast_limits_change(self):
        self.node.clim = self.layer.contrast_limits

    def reset(self):
        self._reset_base()
        self._on_colormap_change()
        self._on_contrast_limits_change()
        self._on_data_change()
