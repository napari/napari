from __future__ import annotations

import numpy as np
from vispy.scene import Node

from napari._vispy.layers.scalar_field import (
    _VISPY_FORMAT_TO_DTYPE,
    ScalarFieldLayerNode,
    VispyScalarFieldBaseLayer,
)
from napari._vispy.layers.tiled_image import TiledImageNode
from napari._vispy.utils.gl import get_gl_extensions, get_max_texture_sizes
from napari._vispy.visuals.image import Image as ImageNode
from napari._vispy.visuals.volume import Volume as VolumeNode
from napari.layers.base._base_constants import Blending
from napari.layers.image.image import Image
from napari.utils.colormaps.colormap_utils import (
    _coerce_contrast_limits,
    _napari_cmap_to_vispy,
)
from napari.utils.translations import trans


class ImageLayerNode(ScalarFieldLayerNode):
    def __init__(
        self, custom_node: Node = None, texture_format: str | None = None
    ) -> None:
        if (
            texture_format == 'auto'
            and 'texture_float' not in get_gl_extensions()
        ):
            # if the GPU doesn't support float textures, texture_format auto
            # WILL fail on float dtypes
            # https://github.com/napari/napari/issues/3988
            texture_format = None

        self._custom_node = custom_node

        (
            self.MAX_TEXTURE_SIZE_2D,
            self.MAX_TEXTURE_SIZE_3D,
        ) = get_max_texture_sizes()

        self._image_node = ImageNode(
            (
                None
                if (texture_format is None or texture_format == 'auto')
                else np.array([[0.0]], dtype=np.float32)
            ),
            method='auto',
            texture_format=texture_format,
        )
        self._tiledimage_node = TiledImageNode(
            np.array([[0.0]], dtype=np.float32),
            tile_size=self.MAX_TEXTURE_SIZE_2D,
            texture_format=texture_format,
        )
        self._volume_node = VolumeNode(
            np.zeros((1, 1, 1), dtype=np.float32),
            clim=[0, 1],
            texture_format=texture_format,
        )

    def get_node(
        self,
        ndisplay: int,
        dtype: np.dtype | None = None,
        shape: tuple | None = None,
    ) -> Node:
        """Return the relevant Vispy VisualNode for current visualization.

        - For small 2D images, this is an Image node.
        - For large 2D images, this is our custom TiledImage node.
        - For 3D images, this is a Volume node.

        Parameters
        ----------
        ndisplay : {2, 3}
            The current number of displayed dimensions.
        dtype : np.dtype
            The dtype of the current data.
        shape : tuple[int, ...]
            The shape of the current data slice.

        Returns:
        node : vispy.scene.Node
            The Node instance to use to display the data.
        """
        # Return custom node if we have one.
        if self._custom_node is not None:
            return self._custom_node

        # Return Image or Volume node based on 2D or 3D.
        M2D = self.MAX_TEXTURE_SIZE_2D
        match ndisplay, shape:
            # 2D grayscale or RGB w/ any dimension exceeding max texture size
            case 2, (s0, s1, *_) if s0 > M2D or s1 > M2D:
                res = self._tiledimage_node
            # any other 2D
            case 2, _:
                res = self._image_node
            # 3D
            case _:
                res = self._volume_node
        if (
            res.texture_format not in {'auto', None}
            and dtype is not None
            and _VISPY_FORMAT_TO_DTYPE[res.texture_format] != dtype
        ):
            # it is a bug to hit this error — it is here to catch bugs
            # early when we are creating the wrong nodes or
            # textures for our data
            raise ValueError(
                trans._(
                    'dtype {dtype} does not match texture_format={texture_format}',
                    dtype=dtype,
                    texture_format=res.texture_format,
                )
            )
        return res


class VispyImageLayer(VispyScalarFieldBaseLayer):
    layer: Image

    def __init__(
        self,
        layer: Image,
        node=None,
        texture_format='auto',
        layer_node_class=ImageLayerNode,
    ) -> None:
        # Track order to detect transpose/roll. Needs to be set before super().__init__()
        self._last_order = None

        super().__init__(
            layer,
            node=node,
            texture_format=texture_format,
            layer_node_class=layer_node_class,
        )

        self.layer.events.interpolation2d.connect(
            self._on_interpolation_change
        )
        self.layer.events.interpolation3d.connect(
            self._on_interpolation_change
        )
        self.layer.events.contrast_limits.connect(
            self._on_contrast_limits_change
        )
        self.layer.events.gamma.connect(self._on_gamma_change)
        self.layer.events.iso_threshold.connect(self._on_iso_threshold_change)
        self.layer.events.attenuation.connect(self._on_attenuation_change)

        # display_change is special (like data_change) because it requires a
        # self.reset(). This means that we have to call it manually. Also,
        # it must be called before reset in order to set the appropriate node
        # first
        self._on_display_change()
        self.reset()
        self._on_data_change()

    def _on_matrix_change(self):
        super()._on_matrix_change()

        # Detect if order changed (transpose or roll)
        current_order = self.layer._slice_input.order
        if current_order != self._last_order:
            if isinstance(self.node, TiledImageNode):
                self.node.handle_axis_change()
            self._last_order = current_order

    def _on_interpolation_change(self) -> None:
        self.node.interpolation = (
            self.layer.interpolation2d
            if self.layer._slice_input.ndisplay == 2
            else self.layer.interpolation3d
        )

    def _on_rendering_change(self) -> None:
        super()._on_rendering_change()
        self._on_attenuation_change()
        self._on_iso_threshold_change()

    def _on_colormap_change(self, event=None) -> None:
        self.node.cmap = _napari_cmap_to_vispy(self.layer.colormap)

    def _update_mip_minip_cutoff(self) -> None:
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

    def _on_contrast_limits_change(self) -> None:
        self.node.clim = _coerce_contrast_limits(
            self.layer.contrast_limits
        ).contrast_limits
        # cutoffs must be updated after clims, so we can set them to the new values
        self._update_mip_minip_cutoff()
        # iso also may depend on contrast limit values
        self._on_iso_threshold_change()

    def _on_blending_change(self, event=None) -> None:
        super()._on_blending_change()
        # cutoffs must be updated after blending, so we can know if
        # the new blending is a translucent one
        self._update_mip_minip_cutoff()

    def _on_gamma_change(self) -> None:
        self.node.gamma = self.layer.gamma

    def _on_iso_threshold_change(self) -> None:
        if isinstance(self.node, VolumeNode):
            if self.node._texture.is_normalized:
                cmin, cmax = self.layer.contrast_limits_range
                self.node.threshold = (self.layer.iso_threshold - cmin) / (
                    cmax - cmin
                )
            else:
                self.node.threshold = self.layer.iso_threshold

    def _on_attenuation_change(self) -> None:
        if isinstance(self.node, VolumeNode):
            self.node.attenuation = self.layer.attenuation

    def reset(self, event=None) -> None:
        super().reset()
        self._on_interpolation_change()
        self._on_colormap_change()
        self._on_contrast_limits_change()
        self._on_gamma_change()
