import warnings
from vispy.scene.visuals import Image as ImageNode
from .volume import Volume as VolumeNode
from vispy.color import Colormap
import numpy as np
from .vispy_base_layer import VispyBaseLayer
from ..layers.image._image_constants import Rendering
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
        node = ImageNode(None, method='auto')
        super().__init__(layer, node)

        # Once #1842 and #1844 from vispy are released and gamma adjustment is
        # done on the GPU these can be dropped
        self._raw_cmap = None
        self._gamma = 1

        # Until we add a specific attenuation parameter to vispy we have to
        # track both iso_threshold and attenuation ourselves.
        self._iso_threshold = 1
        self._attenuation = 1

        self._on_display_change()
        self._on_slice_data_change()

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

    def _on_slice_data_change(self, event=None):
        # Slice data event will be fixed to use passed value after EVH refactor
        # is finished for all layers
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

        # Call to update order of translation values with new dims:
        self._on_scale_change()
        self._on_translate_change()
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
        if isinstance(self.node, VolumeNode):
            self.node.method = rendering
            if Rendering(rendering) == Rendering.ISO:
                self.node.threshold = float(self._iso_threshold)
            elif Rendering(rendering) == Rendering.ATTENUATED_MIP:
                self.node.threshold = float(self._attenuation)

    def _on_colormap_change(self, colormap):
        """Receive layer model colormap change event and update the visual.

        Parameters
        ----------
        colormap : str or tuple
            Colormap name or tuple of (name, vispy.color.Colormap).
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
        # Following should be added to cmap setter in VisPy volume visual
        if isinstance(self.node, VolumeNode):
            self.node.view_program['texture2D_LUT'] = (
                node_cmap.texture_lut()
                if (hasattr(node_cmap, 'texture_lut'))
                else None
            )
        self.node.cmap = node_cmap

    def _on_contrast_limits_change(self, contrast_limits):
        """Receive layer model contrast limits change event and update visual.

        Parameters
        ----------
        contrast_limits : tuple
            Contrast limits.
        """
        # Once #1842 from vispy is released this if else can be dropped
        if isinstance(self.node, VolumeNode):
            self._on_slice_data_change()
        else:
            self.node.clim = contrast_limits

    def _on_gamma_change(self, gamma):
        """Receive the layer model gamma change event and update the visual.

        Parameters
        ----------
        gamma : float
            Gamma value.
        """
        # Once #1842 and #1844 from vispy are released and gamma adjustment is
        # done on the GPU this can be dropped
        if gamma != 1:
            # when gamma!=1, we instantiate a new colormap with 256 control
            # points from 0-1
            cmap = Colormap(self._raw_cmap[np.linspace(0, 1, 256) ** gamma])
        else:
            cmap = self._raw_cmap
        self._gamma = gamma
        # Following should be added to cmap setter in VisPy volume visual
        if isinstance(self.node, VolumeNode):
            self.node.view_program['texture2D_LUT'] = (
                cmap.texture_lut() if (hasattr(cmap, 'texture_lut')) else None
            )
        self.node.cmap = cmap

    def _on_iso_threshold_change(self, iso_threshold):
        """Receive layer model isosurface change event and update the visual.

        Parameters
        ----------
        iso_threshold : float
            Iso surface threshold value, between 0 and 1.
        """
        if (
            isinstance(self.node, VolumeNode)
            and Rendering(self.node.method) == Rendering.ISO
        ):
            self._iso_threshold = iso_threshold
            self.node.threshold = float(iso_threshold)

    def _on_attenuation_change(self, attenuation):
        """Receive layer model attenuation change event and update the visual.

        Parameters
        ----------
        attenuation : float
            Attenuation value, between 0 and 2.
        """
        if (
            isinstance(self.node, VolumeNode)
            and Rendering(self.node.method) == Rendering.ATTENUATED_MIP
        ):
            self._attenuation = attenuation
            self.node.threshold = float(attenuation)

    def reset(self, event=None):
        self._reset_base()
        self._on_colormap_change(self.layer.colormap)
        self._on_rendering_change(self.layer.rendering)
        if isinstance(self.node, ImageNode):
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
