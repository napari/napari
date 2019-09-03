from vispy.scene.visuals import Mesh as MeshNode
from .vispy_base_layer import VispyBaseLayer
import numpy as np


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
        self.layer.dims.events.ndisplay.connect(
            lambda e: self._on_display_change()
        )

        self.reset()
        self._on_display_change()

    def _on_blending_change(self):
        self.node.set_gl_state(
            self.layer.blending, cull_face=False, depth_test=True
        )
        self.node.update()

    def _on_display_change(self):
        self.order = abs(self.node.order)
        self.layer._update_dims()
        self.layer._set_view_slice()
        self.reset()
        for b in range(self.layer.dims.ndisplay):
            self.node.bounds(b)

    def _on_data_change(self):
        if len(self.layer._data_view) == 0 or len(self.layer._view_faces) == 0:
            vertices = None
            faces = None
            values = np.array([0])
        else:
            vertices = self.layer._data_view[:, ::-1] + 0.5
            faces = self.layer._view_faces
            values = self.layer._view_values

        self.node.set_data(
            vertices=vertices, faces=faces, vertex_values=values
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
