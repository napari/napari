import math
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
from vispy.color import Colormap as VispyColormap
from vispy.gloo import Texture2D
from vispy.scene.node import Node

from napari._vispy.layers.image import (
    _DTYPE_TO_VISPY_FORMAT,
    _VISPY_FORMAT_TO_DTYPE,
    ImageLayerNode,
    VispyImageLayer,
    get_dtype_from_vispy_texture_format,
)
from napari._vispy.utils.gl import get_max_texture_sizes
from napari._vispy.visuals.labels import LabelNode
from napari._vispy.visuals.volume import Volume as VolumeNode
from napari.utils.colormaps.colormap import (
    LabelColormap,
    _cast_labels_to_minimum_dtype_auto,
    minimum_dtype_for_labels,
)

if TYPE_CHECKING:
    from napari.layers import Labels


ColorTuple = Tuple[float, float, float, float]


auto_lookup_shader_uint8 = """
uniform sampler2D texture2D_values;

vec4 sample_label_color(float t) {
    if (($use_selection) && ($selection != int(t * 255))) {
        return vec4(0);
    }
    return texture2D(
        texture2D_values,
        vec2(0.0, t)
    );
}
"""

auto_lookup_shader_uint16 = """
uniform sampler2D texture2D_values;

vec4 sample_label_color(float t) {
    // uint 16
    t = t * 65535;
    if (($use_selection) && ($selection != int(t))) {
        return vec4(0);
    }
    float v = mod(t, 256);
    float v2 = (t- v) / 256;
    return texture2D(
        texture2D_values,
        vec2((v + 0.5) / 256, (v2 + 0.5) / 256)
    );
}
"""


direct_lookup_shader = """
uniform sampler2D texture2D_values;
uniform vec2 LUT_shape;

vec4 sample_label_color(float t) {
    t = t * $scale;
    return texture2D(
        texture2D_values,
        vec2(0.0, (t + 0.5) / $color_map_size)
    );
}

"""

direct_lookup_shader_many = """
uniform sampler2D texture2D_values;
uniform vec2 LUT_shape;

vec4 sample_label_color(float t) {
    t = t * $scale;
    float row = mod(t, LUT_shape.x);
    float col = int(t / LUT_shape.x);
    return texture2D(
        texture2D_values,
        vec2((col + 0.5) / LUT_shape.y, (row + 0.5) / LUT_shape.x)
    );
}
"""


class LabelVispyColormap(VispyColormap):
    def __init__(
        self,
        colormap: LabelColormap,
        view_dtype: np.dtype,
        data_dtype: np.dtype,
    ):
        super().__init__(
            colors=["w", "w"], controls=None, interpolation='zero'
        )
        if view_dtype.itemsize == 1:
            shader = auto_lookup_shader_uint8
        elif view_dtype.itemsize == 2:
            shader = auto_lookup_shader_uint16
        else:
            raise ValueError(
                f"Cannot use dtype {view_dtype} with LabelVispyColormap"
            )

        selection = _cast_labels_to_minimum_dtype_auto(
            np.array([colormap.selection]).astype(data_dtype), colormap
        )[0]

        self.glsl_map = (
            shader.replace('$color_map_size', str(len(colormap.colors)))
            .replace('$use_selection', str(colormap.use_selection).lower())
            .replace('$selection', str(selection))
        )


class DirectLabelVispyColormap(VispyColormap):
    def __init__(
        self,
        use_selection=False,
        selection=0.0,
        scale=1.0,
        color_map_size=255,
        multi=False,
    ):
        colors = ['w', 'w']  # dummy values, since we use our own machinery
        super().__init__(colors, controls=None, interpolation='zero')
        shader = direct_lookup_shader_many if multi else direct_lookup_shader
        self.glsl_map = (
            shader.replace("$use_selection", str(use_selection).lower())
            .replace("$selection", str(selection))
            .replace("$scale", str(scale))
            .replace("$color_map_size", str(color_map_size))
        )


def build_textures_from_dict(
    color_dict: Dict[int, ColorTuple], max_size: int
) -> np.ndarray:
    data = np.zeros(
        (
            min(len(color_dict), max_size),
            math.ceil(len(color_dict) / max_size),
            4,
        ),
        dtype=np.float32,
    )
    for key, value in color_dict.items():
        data[key % data.shape[0], key // data.shape[0]] = value
    return data


class VispyLabelsLayer(VispyImageLayer):
    layer: 'Labels'

    def __init__(self, layer, node=None, texture_format='r8') -> None:
        super().__init__(
            layer,
            node=node,
            texture_format=texture_format,
            layer_node_class=LabelLayerNode,
        )

        # self.layer.events.color_mode.connect(self._on_colormap_change)
        self.layer.events.labels_update.connect(self._on_partial_labels_update)
        self.layer.events.selected_label.connect(self._on_colormap_change)
        self.layer.events.show_selected_label.connect(self._on_colormap_change)

    def _on_rendering_change(self):
        # overriding the Image method, so we can maintain the same old rendering name
        if isinstance(self.node, VolumeNode):
            rendering = self.layer.rendering
            self.node.method = (
                rendering
                if rendering != 'translucent'
                else 'translucent_categorical'
            )
            self._on_attenuation_change()
            self._on_iso_threshold_change()

    def _on_colormap_change(self, event=None):
        # self.layer.colormap is a labels_colormap, which is an evented model
        # from napari.utils.colormaps.Colormap (or similar). If we use it
        # in our constructor, we have access to the texture data we need
        if (
            event is not None
            and event.type == 'selected_label'
            and not self.layer.show_selected_label
        ):
            return
        colormap = self.layer.colormap
        mode = self.layer.color_mode
        data_dtype = self.layer._slice.image.view.dtype
        raw_dtype = self.layer._slice.image.raw.dtype
        if mode == 'auto' or data_dtype == raw_dtype:
            if data_dtype != raw_dtype and isinstance(colormap, LabelColormap):
                colormap = LabelColormap(**colormap.dict())
                colormap.background_value = _cast_labels_to_minimum_dtype_auto(
                    np.array([colormap.background_value]).astype(raw_dtype),
                    colormap,
                )[0]
            colors = np.array(
                colormap.map(
                    np.arange(
                        np.iinfo(data_dtype).max + 1, dtype=data_dtype
                    ).reshape(256, -1),
                    apply_selection=False,
                )
            )
            self.node.cmap = LabelVispyColormap(
                colormap, view_dtype=data_dtype, data_dtype=raw_dtype
            )
            self.node.shared_program['texture2D_values'] = Texture2D(
                colors,
                internalformat='rgba32f',
                interpolation='nearest',
            )
            self.texture_data = colors

        elif mode == 'direct':
            color_dict = self.layer._direct_colormap.values_mapping_to_minimum_values_set()[
                1
            ]  # TODO: should probably account for non-given labels
            if isinstance(self.node, VolumeNode):
                max_size = get_max_texture_sizes()[1]
            else:
                max_size = get_max_texture_sizes()[0]
            val_texture = build_textures_from_dict(color_dict, max_size)

            dtype = minimum_dtype_for_labels(
                self.layer._direct_colormap.unique_colors_num() + 2
            )
            if issubclass(dtype.type, np.integer):
                scale = np.iinfo(dtype).max
            else:  # float32 texture
                scale = 1.0

            self.node.cmap = DirectLabelVispyColormap(
                use_selection=colormap.use_selection,
                selection=colormap.selection,
                scale=scale,
                color_map_size=val_texture.shape[0],
                multi=val_texture.shape[1] > 1,
            )
            # note that textures have to be transposed here!
            self.node.shared_program['texture2D_values'] = Texture2D(
                val_texture,  # .swapaxes(0, 1),
                internalformat='rgba32f',
                interpolation='nearest',
            )
            self.node.shared_program['LUT_shape'] = val_texture.shape[:2]
        else:
            self.node.cmap = VispyColormap(*colormap)

    def _on_partial_labels_update(self, event):
        if not self.layer.loaded:
            return

        raw_displayed = self.layer._slice.image.raw
        ndims = len(event.offset)

        if self.node._texture.shape[:ndims] != raw_displayed.shape[:ndims]:
            self.layer.refresh()
            return

        self.node._texture.scale_and_set_data(
            event.data, copy=False, offset=event.offset
        )
        self.node.update()


class LabelLayerNode(ImageLayerNode):
    def __init__(self, custom_node: Node = None, texture_format=None):
        self._custom_node = custom_node
        self._setup_nodes(texture_format)

    def _setup_nodes(self, texture_format):
        self._image_node = LabelNode(
            None
            if (texture_format is None or texture_format == 'auto')
            else np.zeros(
                (1, 1),
                dtype=get_dtype_from_vispy_texture_format(texture_format),
            ),
            method='auto',
            texture_format=texture_format,
        )

        self._volume_node = VolumeNode(
            np.zeros(
                (1, 1, 1),
                dtype=get_dtype_from_vispy_texture_format(texture_format),
            ),
            clim=[0, 2**23 - 1],
            texture_format=texture_format,
        )

    def get_node(self, ndisplay: int, dtype=None) -> Node:
        res = self._image_node if ndisplay == 2 else self._volume_node

        if (
            res.texture_format != "auto"
            and dtype is not None
            and _VISPY_FORMAT_TO_DTYPE[res.texture_format] != dtype
        ):
            self._setup_nodes(_DTYPE_TO_VISPY_FORMAT[dtype])
            return self.get_node(ndisplay, dtype)
        return res
