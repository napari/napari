from vispy.scene.visuals import Mesh
from vispy.color import Colormap
from .vispy_base_layer import VispyBaseLayer
import numpy as np


class VispySurfaceLayer(VispyBaseLayer):
    """Vispy view for the surface layer.

    View is based on the vispy mesh node and uses default values for
    lighting direction and lighting color. More information can be found
    here https://github.com/vispy/vispy/blob/master/vispy/visuals/mesh.py
    """

    def __init__(self, layer):
        node = Mesh()

        super().__init__(layer, node)

        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.contrast_limits.connect(
            self._on_contrast_limits_change
        )
        self.layer.events.gamma.connect(self._on_gamma_change)

        self.reset()
        self._on_data_change()

    def _on_data_change(self, event=None):
        if len(self.layer._data_view) == 0 or len(self.layer._view_faces) == 0:
            vertices = None
            faces = None
            vertex_values = np.array([0])
        else:
            # Offseting so pixels now centered
            vertices = self.layer._data_view[:, ::-1] + 0.5
            faces = self.layer._view_faces
            vertex_values = self.layer._view_vertex_values

        if (
            vertices is not None
            and self.layer.dims.ndisplay == 3
            and self.layer.dims.ndim == 2
        ):
            vertices = np.pad(vertices, ((0, 0), (0, 1)))
        self.node.set_data(
            vertices=vertices, faces=faces, vertex_values=vertex_values
        )
        self.node.update()

    def _on_colormap_change(self, event=None):
        cmap = self.layer.colormap[1]
        if self.layer.gamma != 1:
            # when gamma!=1, we instantiate a new colormap with 256 control
            # points from 0-1
            cmap = Colormap(cmap[np.linspace(0, 1, 256) ** self.layer.gamma])
        if self.layer.dims.ndisplay == 3:
            self.node.view_program['texture2D_LUT'] = (
                cmap.texture_lut() if (hasattr(cmap, 'texture_lut')) else None
            )
        self.node.cmap = cmap

    def _on_contrast_limits_change(self, event=None):
        self.node.clim = self.layer.contrast_limits

    def _on_gamma_change(self, event=None):
        self._on_colormap_change()

    def reset(self, event=None):
        self._reset_base()
        self._on_colormap_change()
        self._on_contrast_limits_change()
