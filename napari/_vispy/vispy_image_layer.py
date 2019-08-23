from vispy.scene.visuals import Image as ImageNode
from vispy.scene.visuals import Volume as VolumeNode
import numpy as np
from .vispy_base_layer import VispyBaseLayer


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

        self._on_display_change()
        self._on_interpolation_change()
        self._on_rendering_change()
        self._on_colormap_change()
        self._on_clim_change()
        self._on_data_change()

    def _on_display_change(self):
        if self.layer.dims.ndisplay == 2 and type(self.node) == VolumeNode:
            parent = self.node.parent
            order = self._order
            self.node.parent = None
            self._order = 0

            self.node = ImageNode(None, method='auto')
            self.node.parent = parent
            self._order = order
            self.layer._position = (0,) * 2
            self.layer._set_view_slice()
            self._on_interpolation_change()
            self._on_clim_change()
            self._on_colormap_change()
            self._on_visible_change()
            self._on_opacity_change()
            self._on_blending_change()
            self._on_scale_change()
            self._on_translate_change()
            self._on_data_change()
        elif self.layer.dims.ndisplay == 3 and type(self.node) == ImageNode:
            parent = self.node.parent
            order = self._order
            self.node.parent = None
            self._order = 0

            self.node = VolumeNode(np.zeros((1, 1, 1)))
            self.node.parent = parent
            self._order = order
            self.layer._position = (0,) * 3
            self.layer._set_view_slice()
            self._on_rendering_change()
            self._on_colormap_change()
            self._on_visible_change()
            self._on_opacity_change()
            self._on_blending_change()
            self._on_scale_change()
            self._on_translate_change()
            self._on_data_change()

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
