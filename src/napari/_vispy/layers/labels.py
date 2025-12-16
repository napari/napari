import math
from typing import TYPE_CHECKING

import numpy as np
from vispy.color import Colormap as VispyColormap
from vispy.gloo import Texture2D
from vispy.scene.node import Node

from napari._vispy.layers.scalar_field import (
    _DTYPE_TO_VISPY_FORMAT,
    _VISPY_FORMAT_TO_DTYPE,
    ScalarFieldLayerNode,
    VispyScalarFieldBaseLayer,
    get_dtype_from_vispy_texture_format,
)
from napari._vispy.utils.gl import get_max_texture_sizes
from napari._vispy.visuals.labels import LabelNode
from napari._vispy.visuals.volume import Volume as VolumeNode
from napari.utils.colormaps.colormap import (
    CyclicLabelColormap,
    _texture_dtype,
)

if TYPE_CHECKING:
    from napari.layers import Labels


ColorTuple = tuple[float, float, float, float]


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
    float v2 = (t - v) / 256;
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
        colormap: CyclicLabelColormap,
        view_dtype: np.dtype,
        raw_dtype: np.dtype,
    ):
        super().__init__(
            colors=['w', 'w'], controls=None, interpolation='zero'
        )
        if view_dtype.itemsize == 1:
            shader = auto_lookup_shader_uint8
        elif view_dtype.itemsize == 2:
            shader = auto_lookup_shader_uint16
        else:
            # See https://github.com/napari/napari/issues/6397
            # Using f32 dtype for textures resulted in very slow fps
            # Therefore, when we have {u,}int{8,16}, we use a texture
            # of that size, but when we have higher bits, we convert
            # to 8-bit on the CPU before sending to the shader.
            # It should thus be impossible to reach this condition.
            raise ValueError(  # pragma: no cover
                f'Cannot use dtype {view_dtype} with LabelVispyColormap'
            )

        selection = colormap._selection_as_minimum_dtype(raw_dtype)

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
            shader.replace('$use_selection', str(use_selection).lower())
            .replace('$selection', str(selection))
            .replace('$scale', str(scale))
            .replace('$color_map_size', str(color_map_size))
        )


def build_textures_from_dict(
    color_dict: dict[int, ColorTuple], max_size: int
) -> np.ndarray:
    """This code assumes that the keys in the color_dict are sequential from 0.

    If any keys are larger than the size of the dictionary, they will
    overwrite earlier keys in the best case, or it might just crash.
    """
    if len(color_dict) > 2**23:
        raise ValueError(  # pragma: no cover
            'Cannot map more than 2**23 colors because of float32 precision. '
            f'Got {len(color_dict)}'
        )
    if len(color_dict) > max_size**2:
        raise ValueError(
            'Cannot create a 2D texture holding more than '
            f'{max_size}**2={max_size**2} colors.'
            f'Got {len(color_dict)}'
        )
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


def _select_colormap_texture(
    colormap: CyclicLabelColormap, view_dtype, raw_dtype
) -> np.ndarray:
    if raw_dtype.itemsize > 2:
        color_texture = colormap._get_mapping_from_cache(view_dtype)
    else:
        color_texture = colormap._get_mapping_from_cache(raw_dtype)

    if color_texture is None:
        raise ValueError(  # pragma: no cover
            f'Cannot build a texture for dtype {raw_dtype=} and {view_dtype=}'
        )
    return color_texture.reshape(256, -1, 4)


class VispyLabelsLayer(VispyScalarFieldBaseLayer):
    layer: 'Labels'

    def __init__(self, layer, node=None, texture_format='r8') -> None:
        super().__init__(
            layer,
            node=node,
            texture_format=texture_format,
            layer_node_class=LabelLayerNode,
        )

        self.layer.events.labels_update.connect(self._on_partial_labels_update)
        self.layer.events.selected_label.connect(self._on_colormap_change)
        self.layer.events.show_selected_label.connect(self._on_colormap_change)
        self.layer.events.iso_gradient_mode.connect(
            self._on_iso_gradient_mode_change
        )
        self.layer.events.data.connect(self._on_colormap_change)
        # as we generate colormap texture based on the data type, we need to
        # update it when the data type changes

    def _on_rendering_change(self):
        # overriding the Image method, so we can maintain the same old rendering name
        if isinstance(self.node, VolumeNode):
            rendering = self.layer.rendering
            self.node.method = (
                rendering
                if rendering != 'translucent'
                else 'translucent_categorical'
            )

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
        auto_mode = isinstance(colormap, CyclicLabelColormap)
        view_dtype = self.layer._slice.image.view.dtype
        raw_dtype = self.layer._slice.image.raw.dtype
        if auto_mode or raw_dtype.itemsize <= 2:
            if raw_dtype.itemsize > 2:
                # If the view dtype is different from the raw dtype, it is possible
                # that background pixels are not the same value as the `background_value`.
                # For example, if raw_dtype is int8 and background_value is `-1`
                # then in view dtype uint8, the background pixels will be 255
                # For data types with more than 16 bits we always cast
                # to uint8 or uint16 and background_value is always 0 in a view array.
                # The LabelColormap is EventedModel, so we need to make
                # a copy instead of temporary overwrite the background_value
                colormap = CyclicLabelColormap(**colormap.dict())
                colormap.background_value = (
                    colormap._background_as_minimum_dtype(raw_dtype)
                )
            color_texture = _select_colormap_texture(
                colormap, view_dtype, raw_dtype
            )
            self.node.cmap = LabelVispyColormap(
                colormap, view_dtype=view_dtype, raw_dtype=raw_dtype
            )
            self.node.shared_program['texture2D_values'] = Texture2D(
                color_texture,
                internalformat='rgba32f',
                interpolation='nearest',
            )
            self.texture_data = color_texture

        elif not auto_mode:  # only for raw_dtype.itemsize > 2
            color_dict = colormap._values_mapping_to_minimum_values_set()[1]
            max_size = get_max_texture_sizes()[0]
            val_texture = build_textures_from_dict(color_dict, max_size)

            dtype = _texture_dtype(
                self.layer._direct_colormap._num_unique_colors + 2,
                raw_dtype,
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
            self.node.shared_program['texture2D_values'] = Texture2D(
                val_texture,
                internalformat='rgba32f',
                interpolation='nearest',
            )
            self.node.shared_program['LUT_shape'] = val_texture.shape[:2]
        else:
            self.node.cmap = VispyColormap(*colormap)

    def _on_iso_gradient_mode_change(self):
        if isinstance(self.node, VolumeNode):
            self.node.iso_gradient_mode = self.layer.iso_gradient_mode

    def _on_partial_labels_update(self, event):
        if not self.layer.loaded:
            return

        raw_displayed = self.layer._slice.image.raw
        ndims = len(event.offset)

        if self.node._texture.shape[:ndims] != raw_displayed.shape[:ndims]:
            # TODO: I'm confused by this whole process; should this refresh be changed?
            self.layer.refresh()
            return

        self.node._texture.scale_and_set_data(
            event.data, copy=False, offset=event.offset
        )
        self.node.update()

    def reset(self, event=None) -> None:
        super().reset()
        self._on_colormap_change()
        self._on_iso_gradient_mode_change()


class LabelLayerNode(ScalarFieldLayerNode):
    def __init__(self, custom_node: Node = None, texture_format=None):
        self._custom_node = custom_node
        self._setup_nodes(texture_format)

    def _setup_nodes(self, texture_format):
        self._image_node = LabelNode(
            (
                None
                if (texture_format is None or texture_format == 'auto')
                else np.zeros(
                    (1, 1),
                    dtype=get_dtype_from_vispy_texture_format(texture_format),
                )
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
            interpolation='nearest',
        )

    def get_node(self, ndisplay: int, dtype=None) -> Node:
        res = self._image_node if ndisplay == 2 else self._volume_node

        if (
            res.texture_format != 'auto'
            and dtype is not None
            and _VISPY_FORMAT_TO_DTYPE[res.texture_format] != dtype
        ):
            self._setup_nodes(_DTYPE_TO_VISPY_FORMAT[dtype])
            return self.get_node(ndisplay, dtype)
        return res
