import numpy as np
from vispy.color import Colormap as VispyColormap
from vispy.scene.visuals import Mesh

from .vispy_base_layer import VispyBaseLayer


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
            # Offsetting so pixels now centered
            vertices = self.layer._data_view[:, ::-1]
            faces = self.layer._view_faces
            vertex_values = self.layer._view_vertex_values

        if (
            vertices is not None
            and self.layer._ndisplay == 3
            and self.layer.ndim == 2
        ):
            vertices = np.pad(vertices, ((0, 0), (0, 1)))
        self.node.set_data(
            vertices=vertices, faces=faces, vertex_values=vertex_values
        )
        self.node.update()
        # Call to update order of translation values with new dims:
        self._on_matrix_change()

    def _on_colormap_change(self, event=None):
        if self.layer.gamma != 1:
            # when gamma!=1, we instantiate a new colormap with 256 control
            # points from 0-1
            colors = self.layer.colormap.map(
                np.linspace(0, 1, 256) ** self.layer.gamma
            )
            cmap = VispyColormap(colors)
        else:
            cmap = VispyColormap(*self.layer.colormap)
        if self.layer._ndisplay == 3:
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
