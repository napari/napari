from vispy.scene.visuals import Image as ImageNode
from vispy.scene.visuals import Volume as VolumeNode
import numpy as np
from .vispy_base_layer import VispyBaseLayer
from ..layers import Image

texture_dtypes = [
    np.dtype(np.int8),
    np.dtype(np.uint8),
    np.dtype(np.int16),
    np.dtype(np.uint16),
    np.dtype(np.float32),
]


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
        self.node.parent = None

        if self.layer.dims.ndisplay == 2:
            self.node = ImageNode(None, method='auto')
        else:
            self.node = VolumeNode(np.zeros((1, 1, 1)))

        self.node.parent = parent
        self.layer._update_dims()
        self.layer._set_view_slice()
        self.reset()

    def _on_data_change(self):
        data = self.layer._data_view
        dtype = np.dtype(data.dtype)
        if dtype not in texture_dtypes:
            try:
                dtype = dict(
                    i=np.int16, f=np.float32, u=np.uint16, b=np.uint8
                )[dtype.kind]
            except KeyError:  # not an int or float
                raise TypeError(
                    f'type {dtype} not allowed for texture; must be one of {set(texture_dtypes)}'
                )
            data = data.astype(dtype)

        if self.layer.dims.ndisplay == 3 and self.layer.dims.ndim == 2:
            data = np.expand_dims(data, axis=0)

        if self.layer.dims.ndisplay == 2:
            self.node._need_colortransform_update = True
            self.node.set_data(data)
        else:
            if dtype == 'float32':
                data = data.copy()
            self.node.set_data(data, clim=self.layer.contrast_limits)
        self.node.update()

    def _on_interpolation_change(self):
        if self.layer.dims.ndisplay == 2:
            self.node.interpolation = self.layer.interpolation

    def _on_rendering_change(self):
        if not self.layer.dims.ndisplay == 2:
            self.node.method = self.layer.rendering

    def _on_colormap_change(self):
        cmap = self.layer.colormap[1]
        # Below is fixed in #1712
        if not self.layer.dims.ndisplay == 2:
            self.node.view_program['texture2D_LUT'] = (
                cmap.texture_lut() if (hasattr(cmap, 'texture_lut')) else None
            )
        self.node.cmap = cmap

    def _on_contrast_limits_change(self):
        if self.layer.dims.ndisplay == 2:
            self.node.clim = self.layer.contrast_limits
        else:
            self._on_data_change()

    def _on_scale_change(self):
        self.scale = [
            self.layer.scale[d] * self.layer._scale_view[d]
            for d in self.layer.dims.displayed[::-1]
        ]
        if self.layer.is_pyramid:
            self.layer.top_left = self.find_top_left()
        self.layer.position = self._transform_position(self._position)

    def _on_translate_change(self):
        self.translate = [
            self.layer.translate[d]
            + self.layer._translate_view[d]
            + self.layer.translate_grid[d]
            for d in self.layer.dims.displayed[::-1]
        ]
        self.layer.position = self._transform_position(self._position)

    def compute_data_level(self, size):
        """Computed what level of the pyramid should be viewed given the
        current size of the requested field of view.

        Parameters
        ----------
        size : 2-tuple
            Requested size of field of view in image coordinates

        Returns
        ----------
        level : int
            Level of the pyramid to be viewing.
        """
        # Convert requested field of view from the camera into log units
        size = np.log2(np.max(size))

        # Max allowed tile in log units
        max_size = np.log2(self.layer._max_tile_shape)

        # Allow for more than 2x coverage of field of view with max tile
        diff = size - max_size + 1.25

        # Find closed downsample level to diff
        ds = self.layer.level_downsamples[:, self.layer.dims.displayed].max(
            axis=1
        )
        level = np.argmin(abs(np.log2(ds) - diff))

        return level

    def find_top_left(self):
        """Finds the top left pixel of the canvas. Depends on the current
        pan and zoom position

        Returns
        ----------
        top_left : tuple of int
            Coordinates of top left pixel.
        """
        nd = self.layer.dims.ndisplay
        # Find image coordinate of top left canvas pixel
        if self.node.canvas is not None:
            transform = self.node.canvas.scene.node_transform(self.node)
            pos = (
                transform.map([0, 0])[:nd]
                + self.translate[:nd] / self.scale[:nd]
            )
        else:
            pos = [0] * nd

        top_left = np.zeros(self.layer.ndim, dtype=int)
        for i, d in enumerate(self.layer.dims.displayed[::-1]):
            top_left[d] = pos[i]

        # Clip according to the max image shape
        top_left = np.clip(
            top_left, 0, np.subtract(self.layer.level_shapes[0], 1)
        )

        # Convert to offset for image array
        rounding_factor = self.layer._max_tile_shape / 4
        top_left = rounding_factor * np.floor(top_left / rounding_factor)

        return top_left.astype(int)

    def on_draw(self, event):
        """Called whenever the canvas is drawn, which happens whenever new
        data is sent to the canvas or the camera is moved.
        """
        self.layer.scale_factor = self.scale_factor
        if self.layer.is_pyramid:
            self.layer.scale_factor = self.scale_factor
            size = self.camera.rect.size
            data_level = self.compute_data_level(size)

            if data_level != self.layer.data_level:
                self.layer.data_level = data_level
            else:
                self.layer.top_left = self.find_top_left()

    def reset(self):
        self._reset_base()
        self._on_interpolation_change()
        self._on_rendering_change()
        self._on_colormap_change()
        self._on_contrast_limits_change()
        self._on_data_change()
