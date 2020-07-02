from vispy.scene.visuals import Mesh
from vispy.color import Colormap
from .vispy_base_layer import VispyBaseLayer
import numpy as np
from ..utils.colormaps import ensure_colormap_tuple


class VispySurfaceLayer(VispyBaseLayer):
    """Vispy view for the surface layer.

    View is based on the vispy mesh node and uses default values for
    lighting direction and lighting color. More information can be found
    here https://github.com/vispy/vispy/blob/master/vispy/visuals/mesh.py
    """

    def __init__(self, layer):
        node = Mesh()

        super().__init__(layer, node)

        self.reset()
        self._on_slice_data_change()

    def _on_slice_data_change(self, event=None):
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
        # Call to update order of translation values with new dims:
        self._on_scale_change()
        self._on_translate_change()

    def _on_colormap_change(self, colormap):
        """Receive layer model colormap change event and update visual.

        Parameters
        ----------
        colormap : tuple
            colormap name and colormap
        """

        name, cmap = ensure_colormap_tuple(colormap)
        # Once #1842 and #1844 from vispy are released and gamma adjustment is
        # done on the GPU this can be dropped
        self._raw_cmap = cmap
        if self._gamma != 1:
            # when gamma!=1, we instantiate a new colormap with 256 control
            # points from 0-1
            node_cmap = Colormap(cmap[np.linspace(0, 1, 256) ** self._gamma])
        else:
            node_cmap = cmap
        self.node.cmap = node_cmap

    def _on_contrast_limits_change(self, contrast_limits):
        """Receive layer model contrast limits change event and update visual.

        Parameters
        ----------
        contrast_limits : tuple
            Contrast limits.
        """

        self.node.clim = contrast_limits

    def _on_gamma_change(self, gamma):
        """Receive layer model gamma change event and update visual.

        Parameters
        ----------
        gamma : float
            New gamma value.
        """

        self._on_colormap_change(self.layer.colormap)

    def reset(self, event=None):
        self._reset_base()
        self._on_colormap_change(self.layer.colormap)
        self._on_contrast_limits_change(self.layer.contrast_limits)
