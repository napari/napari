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
        self.layer.events.contrast_limits.connect(
            lambda e: self._on_contrast_limits_change()
        )
        self.layer.dims.events.ndisplay.connect(
            lambda e: self._on_display_change()
        )

        self._on_display_change()

    def _on_display_change(self):
        parent = self.node.parent
        order = abs(self.node.order)
        self.node.parent = None

        if self.layer.dims.ndisplay == 2:
            self.node = ImageNode(None, method='auto')
        else:
            self.node = VolumeNode(np.zeros((1, 1, 1)))

        self.node.parent = parent
        self.order = order
        self.layer._update_dims()
        self.layer._set_view_slice()
        self.reset()

    def _on_data_change(self):
        if self.layer.dims.ndisplay == 3:
            self.node.set_data(
                self.layer._data_view,
                contrast_limits=self.layer.contrast_limits,
            )
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

    def _on_contrast_limits_change(self):
        if self.layer.dims.ndisplay == 3:
            self.node.set_data(
                self.layer._data_view,
                contrast_limits=self.layer.contrast_limits,
            )
        else:
            self.node.clim = self.layer.contrast_limits

    def reset(self):
        self._reset_base()
        self._on_interpolation_change()
        self._on_rendering_change()
        self._on_colormap_change()
        self._on_contrast_limits_change()
        self._on_data_change()
