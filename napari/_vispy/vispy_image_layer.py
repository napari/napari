from vispy.scene.visuals import Image as ImageNode
from vispy.scene.visuals import Volume as VolumeNode
import numpy as np
from .vispy_base_layer import VispyBaseLayer
from ..layers import Image


class VispyImageLayer(VispyBaseLayer):
    def __init__(self, layer):
        node = ImageNode(None, method='auto')
        super().__init__(layer, node)

        self.layer.events.rendering.connect(
            lambda e: self._on_rendering_change()
        )
        self.layer.events.interpolation.connect(
            lambda e: self._on_interpolation_change()
        )
        self.layer.events.colormap.connect(
            lambda e: self._on_colormap_change()
        )
        self.layer.events.clim.connect(lambda e: self._on_clim_change())
        self.layer.dims.events.display.connect(
            lambda e: self._on_display_change()
        )

        self.reset()

    def _on_display_change(self):
        if self.layer.dims.ndisplay == 2 and isinstance(self.node, VolumeNode):
            parent = self.node.parent
            order = abs(self.node.order)
            self.node.parent = None

            self.node = ImageNode(None, method='auto')
            self.node.parent = parent
            self.order = order
            self.layer._update_dims()
            self.layer._set_view_slice()
            self.reset()

        elif self.layer.dims.ndisplay == 3 and isinstance(
            self.node, ImageNode
        ):
            parent = self.node.parent
            order = abs(self.node.order)
            self.node.parent = None

            self.node = VolumeNode(np.zeros((1, 1, 1)))
            self.node.parent = parent
            self.order = order
            self.layer._update_dims()
            self.layer._set_view_slice()
            self.reset()

    def _on_data_change(self):
        if self.layer.dims.ndisplay == 3:
            self.node.set_data(self.layer._data_view, clim=self.layer.clim)
        else:
            self.node._need_colortransform_update = True
            self.node.set_data(self.layer._data_view)
        self.node.update()

    def _on_interpolation_change(self):
        if self.layer.dims.ndisplay == 2:
            self.node.interpolation = self.layer.interpolation

    def _on_rendering_change(self):
        if self.layer.dims.ndisplay == 3:
            self.node.method = self.layer.rendering

    def _on_colormap_change(self):
        cmap = self.layer.colormap[1]
        if self.layer.dims.ndisplay == 3:
            self.node.view_program['texture2D_LUT'] = (
                cmap.texture_lut() if (hasattr(cmap, 'texture_lut')) else None
            )
        self.node.cmap = cmap

    def _on_clim_change(self):
        if self.layer.dims.ndisplay == 3:
            self.node.set_data(self.layer._data_view, clim=self.layer.clim)
        else:
            self.node.clim = self.layer.clim

    def reset(self):
        self._reset_base()
        self._on_display_change()
        self._on_interpolation_change()
        self._on_rendering_change()
        self._on_colormap_change()
        self._on_clim_change()
        self._on_data_change()
