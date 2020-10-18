import warnings
from typing import Optional

import numpy as np
from vispy.color import Colormap as VispyColormap

from ..image import Image as ImageNode
from ..vispy_base_layer import VispyBaseLayer
from ..volume import Volume as VolumeNode

texture_dtypes = [
    np.dtype(np.int8),
    np.dtype(np.uint8),
    np.dtype(np.int16),
    np.dtype(np.uint16),
    np.dtype(np.float32),
]


def _convert_dtype(data):
    """Convert the data's dtype if necessary.

    Parameters
    ----------
    data
        Convert the dtype of this data.
    """
    dtype = np.dtype(data.dtype)
    if dtype in texture_dtypes:
        return data
    try:
        dtype = dict(i=np.int16, f=np.float32, u=np.uint16, b=np.uint8)[
            dtype.kind
        ]
    except KeyError:  # not an int or float
        raise TypeError(
            f'type {dtype} not allowed for texture; must be one of {set(texture_dtypes)}'  # noqa: E501
        )
    return data.astype(dtype)


class VispyImageLayer2(VispyBaseLayer):
    def __init__(self, layer):
        self._image_node = ImageNode(None, method='auto')
        self._volume_node = VolumeNode(np.zeros((1, 1, 1)), clim=[0, 1])
        super().__init__(layer, self._image_node)
        self._array_like = True

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

        if self.layer.loaded:
            self.node.set_data(data)

        self.node.parent = parent
        self.node.order = self.order
        self.reset()

    def _get_max_texture_size(self) -> Optional[int]:
        """Return maximum size of the texture or None if no max size."""
        ndisplay = self.layer.dims.ndisplay
        if self.MAX_TEXTURE_SIZE_2D is not None and ndisplay == 2:
            return self.MAX_TEXTURE_SIZE_2D
        elif self.MAX_TEXTURE_SIZE_3D is not None and ndisplay == 3:
            return self.MAX_TEXTURE_SIZE_3D
        return None

    def _resize_texture(self, data):
        """Return texture resized if necessary.

        Parameters
        ----------
        data
            Resize this texture if needed.
        """
        max_texture_size = self._get_max_texture_size()
        if max_texture_size is not None:
            return self.downsample_texture(data, self.MAX_TEXTURE_SIZE_2D)
        return data

    def _expand_dims(self, data):
        """Return data with dims expanded if needed.

        Parameters
        ----------
        data
            Expand the dims of this data if needed.
        """
        if self.layer.dims.ndisplay == 3 and self.layer.dims.ndim == 2:
            return np.expand_dims(data, axis=0)
        return data

    def _wrong_visual(self) -> bool:
        """Return whether the current visual is the wrong type.

        bool
            True if the visual is currently the wrong type.
        """
        ndisplay = self.layer.dims.ndisplay
        return (ndisplay == 3 and not isinstance(self.node, VolumeNode)) or (
            ndisplay == 2 and not isinstance(self.node, ImageNode)
        )

    def _on_data_change(self, event=None) -> None:
        """Our self.layer._data_view has been updated, update our node.
        """
        if not self.layer.loaded:
            # Do nothing if we are not yet loaded.
            return

        # Pass the data to our node.
        self._set_new_data(self.layer._data_view, self.node)

    def _set_new_data(self, data, node) -> None:
        """Configure our node to display this new data.

        data
            The new data to display.
        """
        data = _convert_dtype(data)
        data = self._expand_dims(data)
        data = self._resize_texture(data)

        if self._wrong_visual():
            self._on_display_change(data)
        else:
            node.set_data(data)

        self.node.visible = False if self.layer._empty else self.layer.visible

        self._on_scale_change()
        self._on_translate_change()
        node.update()

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
