import warnings
from vispy.scene.visuals import Image as ImageNode
from .volume import Volume as VolumeNode
from vispy.color import Colormap
import numpy as np
from .vispy_base_layer import VispyBaseLayer
from ..layers.image._constants import Rendering
from ..layers import Image, Labels


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

        self.layer.events.rendering.connect(self._on_rendering_change)
        self.layer.events.interpolation.connect(self._on_interpolation_change)
        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.contrast_limits.connect(
            self._on_contrast_limits_change
        )
        self.layer.events.gamma.connect(self._on_gamma_change)
        self.layer.events.iso_threshold.connect(self._on_threshold_change)
        self.layer.events.attenuation.connect(self._on_threshold_change)

        self._on_display_change()
        self._on_data_change()

    def _on_display_change(self, data=None):
        parent = self.node.parent
        self.node.parent = None

        if self.layer.dims.ndisplay == 2:
            self.node = ImageNode(data, method='auto')
        else:
            if data is None:
                data = np.zeros((1, 1, 1))
            self.node = VolumeNode(data, clim=self.layer.contrast_limits)

        self.node.parent = parent
        self.reset()

    def _on_data_change(self, event=None):
        data = self.layer._data_view
        dtype = np.dtype(data.dtype)
        if dtype not in texture_dtypes:
            try:
                dtype = dict(
                    i=np.int16, f=np.float32, u=np.uint16, b=np.uint8
                )[dtype.kind]
            except KeyError:  # not an int or float
                raise TypeError(
                    f'type {dtype} not allowed for texture; must be one of {set(texture_dtypes)}'  # noqa: E501
                )
            data = data.astype(dtype)

        if self.layer.dims.ndisplay == 3 and self.layer.dims.ndim == 2:
            data = np.expand_dims(data, axis=0)

        # Check if data exceeds MAX_TEXTURE_SIZE and downsample
        if (
            self.MAX_TEXTURE_SIZE_2D is not None
            and self.layer.dims.ndisplay == 2
        ):
            data = self.downsample_texture(data, self.MAX_TEXTURE_SIZE_2D)
        elif (
            self.MAX_TEXTURE_SIZE_3D is not None
            and self.layer.dims.ndisplay == 3
        ):
            data = self.downsample_texture(data, self.MAX_TEXTURE_SIZE_3D)

        # Check if ndisplay has changed current node type needs updating
        if (
            self.layer.dims.ndisplay == 3
            and not isinstance(self.node, VolumeNode)
        ) or (
            self.layer.dims.ndisplay == 2
            and not isinstance(self.node, ImageNode)
        ):
            self._on_display_change(data)
        else:
            if self.layer.dims.ndisplay == 2:
                self.node._need_colortransform_update = True
                self.node.set_data(data)
            else:
                self.node.set_data(data, clim=self.layer.contrast_limits)
        self.node.update()

    def _on_interpolation_change(self, event=None):
        if self.layer.dims.ndisplay == 3 and isinstance(self.layer, Labels):
            self.node.interpolation = 'nearest'
        elif self.layer.dims.ndisplay == 3 and isinstance(self.layer, Image):
            self.node.interpolation = 'linear'
        else:
            self.node.interpolation = self.layer.interpolation

    def _on_rendering_change(self, event=None):
        if self.layer.dims.ndisplay == 3:
            self.node.method = self.layer.rendering
            self._on_threshold_change()

    def _on_colormap_change(self, event=None):
        cmap = self.layer.colormap[1]
        if self.layer.gamma != 1:
            # when gamma!=1, we instantiate a new colormap
            # with 256 control points from 0-1
            cmap = Colormap(cmap[np.linspace(0, 1, 256) ** self.layer.gamma])

        # Below is fixed in #1712
        if not self.layer.dims.ndisplay == 2:
            self.node.view_program['texture2D_LUT'] = (
                cmap.texture_lut() if (hasattr(cmap, 'texture_lut')) else None
            )
        self.node.cmap = cmap

    def _on_contrast_limits_change(self, event=None):
        if self.layer.dims.ndisplay == 2:
            self.node.clim = self.layer.contrast_limits
        else:
            self._on_data_change()

    def _on_gamma_change(self, event=None):
        self._on_colormap_change()

    def _on_threshold_change(self, event=None):
        if self.layer.dims.ndisplay == 2:
            return
        rendering = self.layer.rendering
        if isinstance(rendering, str):
            rendering = Rendering(rendering)
        if rendering == Rendering.ISO:
            self.node.threshold = float(self.layer.iso_threshold)
        elif rendering == Rendering.ATTENUATED_MIP:
            self.node.threshold = float(self.layer.attenuation)

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

    def reset(self, event=None):
        self._reset_base()
        self._on_interpolation_change()
        self._on_colormap_change()
        self._on_rendering_change()
        if self.layer.dims.ndisplay == 2:
            self._on_contrast_limits_change()

    def downsample_texture(self, data, MAX_TEXTURE_SIZE):
        """Downsample data based on maximum allowed texture size.

        Parameters
        ----------
        data : array
            Data to be downsampled if needed.
        MAX_TEXTURE_SIZE : int
            Maximum allowed texture size.

        Returns
        -------
        data : array
            Data that now fits inside texture.
        """
        if np.any(np.greater(data.shape, MAX_TEXTURE_SIZE)):
            if self.layer.is_pyramid:
                raise ValueError(
                    f"Shape of individual tiles in pyramid {data.shape} "
                    f"cannot exceed GL_MAX_TEXTURE_SIZE "
                    f"{MAX_TEXTURE_SIZE}. The max tile shape "
                    f"`layer._max_tile_shape` {self.layer._max_tile_shape}"
                    f" must be reduced. Rendering is currently in "
                    f"{self.layer.dims.ndisplay}D mode."
                )
            warnings.warn(
                f"data shape {data.shape} exceeds GL_MAX_TEXTURE_SIZE "
                f"{MAX_TEXTURE_SIZE} in at least one axis and "
                f"will be downsampled. Rendering is currently in "
                f"{self.layer.dims.ndisplay}D mode."
            )
            downsample = np.ceil(
                np.divide(data.shape, MAX_TEXTURE_SIZE)
            ).astype(int)
            scale = np.ones(self.layer.ndim)
            for i, d in enumerate(self.layer.dims.displayed):
                scale[d] = downsample[i]
            self.layer._scale_view = scale
            self._on_scale_change()
            slices = tuple(slice(None, None, ds) for ds in downsample)
            data = data[slices]
        return data
