import warnings

import numpy as np
from vispy.color import Colormap as VispyColormap
from vispy.scene.node import Node

from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.utils.gl import fix_data_dtype, get_gl_extensions
from napari._vispy.visuals.image import Image as ImageNode
from napari._vispy.visuals.volume import Volume as VolumeNode
from napari.layers.base._base_constants import Blending
from napari.utils.translations import trans


class ImageLayerNode:
    def __init__(self, custom_node: Node = None, texture_format=None) -> None:
        if (
            texture_format == 'auto'
            and 'texture_float' not in get_gl_extensions()
        ):
            # if the GPU doesn't support float textures, texture_format auto
            # WILL fail on float dtypes
            # https://github.com/napari/napari/issues/3988
            texture_format = None

        self._custom_node = custom_node
        self._image_node = ImageNode(
            None,
            method='auto',
            texture_format=texture_format,
        )
        self._volume_node = VolumeNode(
            np.zeros((1, 1, 1), dtype=np.float32),
            clim=[0, 1],
            texture_format=texture_format,
        )

    def get_node(self, ndisplay: int) -> Node:
        # Return custom node if we have one.
        if self._custom_node is not None:
            return self._custom_node

        # Return Image or Volume node based on 2D or 3D.
        if ndisplay == 2:
            return self._image_node
        return self._volume_node


class VispyImageLayer(VispyBaseLayer):
    def __init__(self, layer, node=None, texture_format='auto') -> None:
        # Use custom node from caller, or our standard image/volume nodes.
        self._layer_node = ImageLayerNode(node, texture_format=texture_format)

        # Default to 2D (image) node.
        super().__init__(layer, self._layer_node.get_node(2))

        self._array_like = True

        self.layer.events.rendering.connect(self._on_rendering_change)
        self.layer.events.depiction.connect(self._on_depiction_change)
        self.layer.events.interpolation2d.connect(
            self._on_interpolation_change
        )
        self.layer.events.interpolation3d.connect(
            self._on_interpolation_change
        )
        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.contrast_limits.connect(
            self._on_contrast_limits_change
        )
        self.layer.events.gamma.connect(self._on_gamma_change)
        self.layer.events.iso_threshold.connect(self._on_iso_threshold_change)
        self.layer.events.attenuation.connect(self._on_attenuation_change)
        self.layer.plane.events.position.connect(
            self._on_plane_position_change
        )
        self.layer.plane.events.thickness.connect(
            self._on_plane_thickness_change
        )
        self.layer.plane.events.normal.connect(self._on_plane_normal_change)
        self.layer.events.custom_interpolation_kernel_2d.connect(
            self._on_custom_interpolation_kernel_2d_change
        )

        # display_change is special (like data_change) because it requires a self.reset()
        # this means that we have to call it manually. Also, it must be called before reset
        # in order to set the appropriate node first
        self._on_display_change()
        self.reset()
        self._on_data_change()

    def _on_display_change(self, data=None):
        parent = self.node.parent
        self.node.parent = None
        ndisplay = self.layer._slice_input.ndisplay
        self.node = self._layer_node.get_node(ndisplay)

        if data is None:
            data = np.zeros((1,) * ndisplay, dtype=np.float32)

        self.node.visible = self.layer.visible

        self.node.set_data(data)

        self.node.parent = parent
        self.node.order = self.order
        for overlay_visual in self.overlays.values():
            overlay_visual.node.parent = self.node
        self.reset()

    def _on_data_change(self):
        node = self.node
        data = fix_data_dtype(self.layer._data_view)
        ndisplay = self.layer._slice_input.ndisplay

        if ndisplay == 3 and self.layer.ndim == 2:
            data = np.expand_dims(data, axis=0)

        # Check if data exceeds MAX_TEXTURE_SIZE and downsample
        if self.MAX_TEXTURE_SIZE_2D is not None and ndisplay == 2:
            data = self.downsample_texture(data, self.MAX_TEXTURE_SIZE_2D)
        elif self.MAX_TEXTURE_SIZE_3D is not None and ndisplay == 3:
            data = self.downsample_texture(data, self.MAX_TEXTURE_SIZE_3D)

        # Check if ndisplay has changed current node type needs updating
        if (ndisplay == 3 and not isinstance(node, VolumeNode)) or (
            ndisplay == 2 and not isinstance(node, ImageNode)
        ):
            self._on_display_change(data)
        else:
            node.set_data(data)

        # Call to update order of translation values with new dims:
        self._on_matrix_change()
        node.update()

    def _on_interpolation_change(self):
        self.node.interpolation = (
            self.layer.interpolation2d
            if self.layer._slice_input.ndisplay == 2
            else self.layer.interpolation3d
        )

    def _on_custom_interpolation_kernel_2d_change(self):
        if self.layer._slice_input.ndisplay == 2:
            self.node.custom_kernel = self.layer.custom_interpolation_kernel_2d

    def _on_rendering_change(self):
        if isinstance(self.node, VolumeNode):
            self.node.method = self.layer.rendering
            self._on_attenuation_change()
            self._on_iso_threshold_change()

    def _on_depiction_change(self):
        if isinstance(self.node, VolumeNode):
            self.node.raycasting_mode = str(self.layer.depiction)

    def _on_colormap_change(self):
        self.node.cmap = VispyColormap(*self.layer.colormap)

    def _update_mip_minip_cutoff(self):
        # discard fragments beyond contrast limits, but only with translucent blending
        if isinstance(self.node, VolumeNode):
            if self.layer.blending in {
                Blending.TRANSLUCENT,
                Blending.TRANSLUCENT_NO_DEPTH,
            }:
                self.node.mip_cutoff = self.node._texture.clim_normalized[0]
                self.node.minip_cutoff = self.node._texture.clim_normalized[1]
            else:
                self.node.mip_cutoff = None
                self.node.minip_cutoff = None

    def _on_contrast_limits_change(self):
        self.node.clim = self.layer.contrast_limits
        # cutoffs must be updated after clims, so we can set them to the new values
        self._update_mip_minip_cutoff()
        # iso also may depend on contrast limit values
        self._on_iso_threshold_change()

    def _on_blending_change(self):
        super()._on_blending_change()
        # cutoffs must be updated after blending, so we can know if
        # the new blending is a translucent one
        self._update_mip_minip_cutoff()

    def _on_gamma_change(self):
        if len(self.node.shared_program.frag._set_items) > 0:
            self.node.gamma = self.layer.gamma

    def _on_iso_threshold_change(self):
        if isinstance(self.node, VolumeNode):
            if self.node._texture.is_normalized:
                cmin, cmax = self.layer.contrast_limits_range
                self.node.threshold = (self.layer.iso_threshold - cmin) / (
                    cmax - cmin
                )
            else:
                self.node.threshold = self.layer.iso_threshold

    def _on_attenuation_change(self):
        if isinstance(self.node, VolumeNode):
            self.node.attenuation = self.layer.attenuation

    def _on_plane_thickness_change(self):
        if isinstance(self.node, VolumeNode):
            self.node.plane_thickness = self.layer.plane.thickness

    def _on_plane_position_change(self):
        if isinstance(self.node, VolumeNode):
            self.node.plane_position = self.layer.plane.position

    def _on_plane_normal_change(self):
        if isinstance(self.node, VolumeNode):
            self.node.plane_normal = self.layer.plane.normal

    def reset(self, event=None):
        super().reset()
        self._on_interpolation_change()
        self._on_colormap_change()
        self._on_contrast_limits_change()
        self._on_gamma_change()
        self._on_rendering_change()
        self._on_depiction_change()
        self._on_plane_position_change()
        self._on_plane_normal_change()
        self._on_plane_thickness_change()
        self._on_custom_interpolation_kernel_2d_change()

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
                    trans._(
                        "Shape of in dividual tiles in multiscale {shape} cannot exceed GL_MAX_TEXTURE_SIZE {texture_size}. Rendering is currently in {ndisplay}D mode.",
                        deferred=True,
                        shape=data.shape,
                        texture_size=MAX_TEXTURE_SIZE,
                        ndisplay=self.layer._slice_input.ndisplay,
                    )
                )
            warnings.warn(
                trans._(
                    "data shape {shape} exceeds GL_MAX_TEXTURE_SIZE {texture_size} in at least one axis and will be downsampled. Rendering is currently in {ndisplay}D mode.",
                    deferred=True,
                    shape=data.shape,
                    texture_size=MAX_TEXTURE_SIZE,
                    ndisplay=self.layer._slice_input.ndisplay,
                )
            )
            downsample = np.ceil(
                np.divide(data.shape, MAX_TEXTURE_SIZE)
            ).astype(int)
            scale = np.ones(self.layer.ndim)
            for i, d in enumerate(self.layer._slice_input.displayed):
                scale[d] = downsample[i]
            self.layer._transforms['tile2data'].scale = scale
            self._on_matrix_change()
            slices = tuple(slice(None, None, ds) for ds in downsample)
            data = data[slices]
        return data
