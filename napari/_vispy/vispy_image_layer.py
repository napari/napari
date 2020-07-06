import warnings
from .volume import Volume as VolumeNode
import numpy as np
from .vispy_base_layer import VispyBaseLayer
from ..utils.colormaps import ensure_colormap_tuple


texture_dtypes = [
    np.dtype(np.int8),
    np.dtype(np.uint8),
    np.dtype(np.int16),
    np.dtype(np.uint16),
    np.dtype(np.float32),
]


class VispyImageLayer(VispyBaseLayer):
    def __init__(self, layer):
        super().__init__(layer, VolumeNode(np.zeros((1, 1, 1))))

        self._on_slice_data_change()
        self.reset()

    def _on_slice_data_change(self, event=None):
        # Slice data event will be fixed to use passed value after EVH refactor
        # is finished for all layers
        data = self.layer._data_view

        # Make sure data is correct dtype
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

        # RGB data is not currently supported for the volume visual
        if self.layer.rgb:
            data = data.mean(axis=-1)

        # If ndisplay is two add an axis
        if self.layer.dims.ndisplay == 2:
            data = np.expand_dims(data, axis=0)

        self.node.set_data(data)
        self.node.update()

    def _on_interpolation_change(self, interpolation):
        """Receive layer model isosurface change event and update the visual.

        Parameters
        ----------
        interpolation : float
            Iso surface threshold value, between 0 and 1.
        """
        self.node.interpolation = interpolation

    def _on_rendering_change(self, rendering):
        """Receive layer model rendering change event and update dropdown menu.

        Parameters
        ----------
        text : str
            Rendering mode used by VisPy.
            Selects a preset rendering mode in VisPy that determines how
            volume is displayed:
            * translucent: voxel colors are blended along the view ray until
              the result is opaque.
            * mip: maxiumum intensity projection. Cast a ray and display the
              maximum value that was encountered.
            * additive: voxel colors are added along the view ray until
              the result is saturated.
            * iso: isosurface. Cast a ray until a certain threshold is
              encountered. At that location, lighning calculations are
              performed to give the visual appearance of a surface.
            * attenuated_mip: attenuated maxiumum intensity projection. Cast a
              ray and attenuate values based on integral of encountered values,
              display the maximum value that was encountered after attenuation.
              This will make nearer objects appear more prominent.
        """
        self.node.method = rendering

    def _on_colormap_change(self, colormap):
        """Receive layer model colormap change event and update the visual.

        Parameters
        ----------
        colormap : str or tuple
            Colormap name or tuple of (name, vispy.color.Colormap).
        """
        name, cmap = ensure_colormap_tuple(colormap)
        self.node.cmap = cmap

    def _on_contrast_limits_change(self, contrast_limits):
        """Receive layer model contrast limits change event and update visual.

        Parameters
        ----------
        contrast_limits : tuple
            Contrast limits.
        """
        self.node.clim = contrast_limits

    def _on_gamma_change(self, gamma):
        """Receive the layer model gamma change event and update the visual.

        Parameters
        ----------
        gamma : float
            Gamma value.
        """
        self.node.gamma = gamma

    def _on_iso_threshold_change(self, iso_threshold):
        """Receive layer model isosurface change event and update the visual.

        Parameters
        ----------
        iso_threshold : float
            Iso surface threshold value, between 0 and 1.
        """
        self.node.threshold = iso_threshold

    def _on_attenuation_change(self, attenuation):
        """Receive layer model attenuation change event and update the visual.

        Parameters
        ----------
        attenuation : float
            Attenuation value, between 0 and 2.
        """
        self.node.attenuation = attenuation

    def reset(self):
        self._reset_base()
        self._on_colormap_change(self.layer.colormap)
        self._on_rendering_change(self.layer.rendering)
        self._on_iso_threshold_change(self.layer.iso_threshold)
        self._on_attenuation_change(self.layer.attenuation)
        self._on_contrast_limits_change(self.layer.contrast_limits)

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
