import warnings

import numpy as np
from vispy.color import Colormap as VispyColormap

from .image import Image as ImageNode
from .vispy_base_layer import VispyBaseLayer
from .volume import Volume as VolumeNode

texture_dtypes = [
    np.dtype(np.int8),
    np.dtype(np.uint8),
    np.dtype(np.int16),
    np.dtype(np.uint16),
    np.dtype(np.float32),
]


class VispyImageLayer(VispyBaseLayer):
    def __init__(self, layer):
        self._image_node = ImageNode(None, method='auto')
        self._volume_node = VolumeNode(np.zeros((1, 1, 1)), clim=[0, 1])
        super().__init__(layer, self._image_node)

        self.layer.events.rendering.connect(self._on_rendering_change)
        self.layer.events.interpolation.connect(self._on_interpolation_change)
        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.contrast_limits.connect(
            self._on_contrast_limits_change
        )
        self.layer.events.gamma.connect(self._on_gamma_change)
        self.layer.events.iso_threshold.connect(self._on_iso_threshold_change)
        self.layer.events.attenuation.connect(self._on_attenuation_change)

        self._on_display_change()
        self._on_data_change()

    def _on_display_change(self, data=None):
        parent = self.node.parent
        self.node.parent = None

        if self.layer.dims.ndisplay == 2:
            self.node = self._image_node
        else:
            self.node = self._volume_node

        if data is None:
            data = np.zeros((1,) * self.layer.dims.ndisplay)

        if self.layer._empty:
            self.node.visible = False
        else:
            self.node.visible = self.layer.visible

        self.node.set_data(data)
        self.node.parent = parent
        self.node.order = self.order
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
            self.node.set_data(data)

        if self.layer._empty:
            self.node.visible = False
        else:
            self.node.visible = self.layer.visible

        # Call to update order of translation values with new dims:
        self._on_scale_change()
        self._on_translate_change()
        self.node.update()

    def _on_interpolation_change(self, event=None):
        self.node.interpolation = self.layer.interpolation

    def _on_rendering_change(self, event=None):
        if isinstance(self.node, VolumeNode):
            self.node.method = self.layer.rendering
            self._on_attenuation_change()
            self._on_iso_threshold_change()

    def _on_colormap_change(self, event=None):
        self.node.cmap = VispyColormap(*self.layer.colormap)

    def _on_contrast_limits_change(self, event=None):
        self.node.clim = self.layer.contrast_limits

    def _on_gamma_change(self, event=None):
        if len(self.node.shared_program.frag._set_items) > 0:
            self.node.gamma = self.layer.gamma

    def _on_iso_threshold_change(self, event=None):
        if isinstance(self.node, VolumeNode):
            self.node.threshold = self.layer.iso_threshold

    def _on_attenuation_change(self, event=None):
        if isinstance(self.node, VolumeNode):
            self.node.attenuation = self.layer.attenuation

    def reset(self, event=None):
        self._reset_base()
        self._on_interpolation_change()
        self._on_colormap_change()
        self._on_contrast_limits_change()
        self._on_gamma_change()
        self._on_rendering_change()

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
            if self.layer.multiscale:
                raise ValueError(
                    f"Shape of individual tiles in multiscale {data.shape} "
                    f"cannot exceed GL_MAX_TEXTURE_SIZE "
                    f"{MAX_TEXTURE_SIZE}. Rendering is currently in "
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
            self.layer._transforms['tile2data'].scale = scale
            self._on_scale_change()
            slices = tuple(slice(None, None, ds) for ds in downsample)
            data = data[slices]
        return data
