import numpy as np
from vispy.color import Colormap as VispyColormap
from vispy.scene.node import Node
from vispy.scene.visuals import create_visual_node
from vispy.visuals.image import ImageVisual
from vispy.visuals.shaders import Function, FunctionChain

from napari._vispy.layers.image import ImageLayerNode, VispyImageLayer
from napari._vispy.visuals.volume import Volume as VolumeNode

# from napari._vispy.layers.base import VispyBaseLayer


low_disc_lookup_shader = """
uniform sampler2D texture2D_LUT;

vec4 sample_label_color(float t) {
    float phi_mod = 0.6180339887498948482;  // phi - 1
    float value = 0.0;
    float margin = 1.0 / 256;

    if (t == 0) {
        return vec4(0.0,0.0,0.0,0.0);
    }

    if (($use_selection) && ($selection != t)) {
        return vec4(0.0,0.0,0.0,0.0);
    }

    value = mod((t * phi_mod + $seed), 1.0) * (1 - 2*margin) + margin;

    return texture2D(
        texture2D_LUT,
        vec2(0.0, clamp(value, 0.0, 1.0))
    );
}
"""


direct_lookup_shader = """
uniform sampler2D texture2D_keys;
uniform sampler2D texture2D_values;
uniform vec2 LUT_shape;


vec4 sample_label_color(float t) {
    vec2 pos = vec2(mod(t, LUT_shape.x), mod(t, LUT_shape.y));
    vec2 pos_tex = pos / LUT_shape;
    float found = texture2D(
        texture2D_keys,
        pos_tex
    ).r;

    while (abs(found - t) < 0.5) {
        t = t + 1;
        pos = vec2(mod(t, LUT_shape.x), mod(t, LUT_shape.y));
        pos_tex = pos / LUT_shape;
        found = texture2D(
            texture2D_keys,
            pos_tex
        ).r;
    }

    return vec4(pos_tex, 0, 1); // debug if texel is calculated correctly

    vec4 color = texture2D(
        texture2D_values,
        pos_tex
    );
    return color;
}

"""


class LabelVispyColormap(VispyColormap):
    def __init__(
        self,
        colors,
        controls=None,
        seed=0.5,
        use_selection=False,
        selection=0.0,
    ):
        super().__init__(colors, controls, interpolation='zero')
        self.glsl_map = (
            low_disc_lookup_shader.replace('$seed', str(seed))
            .replace('$use_selection', str(use_selection).lower())
            .replace('$selection', str(selection))
        )


class DirectLabelVispyColormap(VispyColormap):
    def __init__(
        self,
        use_selection=False,
        selection=0.0,
    ):
        colors = ['w', 'w']  # dummy values, since we use our own machinery
        super().__init__(colors, controls=None, interpolation='zero')
        self.glsl_map = direct_lookup_shader.replace(
            '$use_selection', str(use_selection).lower()
        ).replace('$selection', str(selection))


def idx_to_2D(idx, shape):
    return (idx % shape[0], idx % shape[1])


def hash2d_get(key, keys, values, empty_val=0):
    pos = idx_to_2D(key, keys.shape)
    initial_key = key
    while keys[pos] != key:
        if key - initial_key > keys.size:
            raise KeyError('label does not exist')
        key += 1
        pos = idx_to_2D(key, keys.shape)
    if keys[pos] == key:
        return pos, values[pos]
    else:
        return None, None


def hash2d_set(key, value, keys, values, empty_val=0):
    if key is None:
        return
    pos = idx_to_2D(key, keys.shape)
    initial_key = key
    while keys[pos] != empty_val:
        if key - initial_key > keys.size:
            raise OverflowError('too many labels')
        pos += 1
        pos = idx_to_2D(key, keys.shape)
    keys[pos] = key
    values[pos] = value


def build_textures_from_dict(color_dict, empty_val=0, shape=(4, 5)):
    keys = np.full(shape, empty_val, dtype=np.float32)
    values = np.zeros(shape + (4,), dtype=np.float32)
    for key, value in color_dict.items():
        hash2d_set(key, value, keys, values)
    return keys, values


class VispyLabelsLayer(VispyImageLayer):
    def __init__(self, layer, node=None, texture_format='r32f'):
        super().__init__(
            layer,
            node=node,
            texture_format=texture_format,
            layer_node_class=LabelLayerNode,
        )

        self.layer.events.color_mode.connect(self._on_colormap_change)

    def _on_colormap_change(self, event=None):
        # self.layer.colormap is a labels_colormap, which is an evented model
        # from napari.utils.colormaps.Colormap (or similar). If we use it
        # in our constructor, we have access to the texture data we need
        colormap = self.layer.colormap
        mode = self.layer.color_mode

        if mode == 'auto':
            self.node.cmap = LabelVispyColormap(
                colors=colormap.colors,
                controls=colormap.controls,
                seed=colormap.seed,
                use_selection=colormap.use_selection,
                selection=colormap.selection,
            )
        elif mode == 'direct':
            color_dict = (
                self.layer.color
            )  # TODO: should probably account for non-given labels
            key_texture, val_texture = build_textures_from_dict(color_dict)
            self.node.cmap = DirectLabelVispyColormap(
                use_selection=colormap.use_selection,
                selection=colormap.selection,
            )
            self.node.shared_program['texture2D_keys'] = key_texture
            self.node.shared_program['texture2D_values'] = val_texture
            self.node.shared_program['LUT_shape'] = key_texture.shape
        else:
            self.node.cmap = VispyColormap(*colormap)


class LabelVisual(ImageVisual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_color_transform(self):
        fun = FunctionChain(
            None,
            [
                Function(self._func_templates['red_to_luminance']),
                Function(self.cmap.glsl_map),
            ],
        )
        return fun


class LabelLayerNode(ImageLayerNode):
    def __init__(self, custom_node: Node = None, texture_format=None):
        self._custom_node = custom_node
        self._image_node = LabelNode(
            None
            if (texture_format is None or texture_format == 'auto')
            else np.array([[0.0]], dtype=np.float32),
            method='auto',
            texture_format=texture_format,
        )

        self._volume_node = VolumeNode(
            np.zeros((1, 1, 1), dtype=np.float32),
            clim=[0, 2**23 - 1],
            texture_format=texture_format,
        )


BaseLabel = create_visual_node(LabelVisual)


class LabelNode(BaseLabel):
    def _compute_bounds(self, axis, view):
        if self._data is None:
            return None
        elif axis > 1:
            return (0, 0)
        else:
            return (0, self.size[axis])
