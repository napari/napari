from math import isnan

import numpy as np
from vispy.color import Colormap as VispyColormap
from vispy.gloo import Texture2D
from vispy.scene.node import Node
from vispy.scene.visuals import create_visual_node
from vispy.visuals.image import ImageVisual
from vispy.visuals.shaders import Function, FunctionChain

from napari._vispy.layers.image import ImageLayerNode, VispyImageLayer
from napari._vispy.utils.gl import get_max_texture_sizes
from napari._vispy.visuals.volume import Volume as VolumeNode

PRIME_NUM_TABLE = [61, 127, 251, 509, 1021, 2039, 4093, 8191, 16381, 32749]

MAX_TEXTURE_SIZE = None

low_disc_lookup_shader = """
uniform sampler2D texture2D_LUT;

vec4 sample_label_color(float t) {
    float phi_mod = 0.6180339887498948482;  // phi - 1
    float value = 0.0;
    float margin = 1.0 / 256;

    if (t == 0) {
        return vec4(0);
    }

    if (($use_selection) && ($selection != t)) {
        return vec4(0);
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
uniform int color_count;


vec4 sample_label_color(float t) {
    if (($use_selection) && ($selection != t)) {
        return vec4(0);
    }

    float empty = 0.;
    // get position in the texture grid (same as hash2d_get)
    vec2 pos = vec2(
        mod(int(t / LUT_shape.y), LUT_shape.x),
        mod(t, LUT_shape.y)
    );

    // add .5 to move to the center of each texel and convert to texture coords
    vec2 pos_tex = (pos + vec2(.5)) / LUT_shape;

    // sample key texture
    float found = texture2D(
        texture2D_keys,
        pos_tex
    ).r;

    // return vec4(pos_tex, 0, 1); // debug if texel is calculated correctly (correct)
    // return vec4(found / 15, 0, 0, 1); // debug if key is calculated correctly (correct, should be a black-to-red gradient)

    // we get a different value:
    // - if it's the empty key, exit;
    // - otherwise, it's a hash collision: continue searching
    float initial_t = t;
    int count = 0;
    while ((abs(found - initial_t) > 1e-8) && (abs(found - empty) > 1e-8)) {
        count = count + 1;
        t = initial_t + float(count);
        if (count >= color_count) {
            return vec4(0);
        }
        // same as above
        vec2 pos = vec2(
            mod(int(t / LUT_shape.y), LUT_shape.x),
            mod(t, LUT_shape.y)
        );
        pos_tex = (pos + vec2(.5)) / LUT_shape;

        found = texture2D(
            texture2D_keys,
            pos_tex
        ).r;
    }

    // return vec4(pos_tex, 0, 1); // debug if final texel is calculated correctly

    vec4 color = vec4(0);
    if (abs(found - empty) > 1e-8) {
        color = texture2D(
            texture2D_values,
            pos_tex
        );
    }
    return color;
}

"""


direct_lookup_shade_without_collision = """
uniform sampler2D texture2D_keys;
uniform sampler2D texture2D_values;
uniform vec2 LUT_shape;
uniform int color_count;


vec4 sample_label_color(float t) {
    if (($use_selection) && ($selection != t)) {
        return vec4(0);
    }

    float empty = 0.;
    // get position in the texture grid (same as hash2d_get)
    vec2 pos = vec2(
        mod(int(t / LUT_shape.y), LUT_shape.x),
        mod(t, LUT_shape.y)
    );

    // add .5 to move to the center of each texel and convert to texture coords
    vec2 pos_tex = (pos + vec2(.5)) / LUT_shape;

    // sample key texture
    float found = texture2D(
        texture2D_keys,
        pos_tex
    ).r;

    // return vec4(pos_tex, 0, 1); // debug if texel is calculated correctly (correct)
    // return vec4(found / 15, 0, 0, 1); // debug if key is calculated correctly (correct, should be a black-to-red gradient)

    if (abs(found - t) > 1e-8) {
        return vec4(0);
    }

    return texture2D(
        texture2D_values,
        pos_tex
    );
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
        collision=True,
    ):
        colors = ['w', 'w']  # dummy values, since we use our own machinery
        super().__init__(colors, controls=None, interpolation='zero')
        if collision:
            self.glsl_map = direct_lookup_shader.replace(
                '$use_selection', str(use_selection).lower()
            ).replace('$selection', str(selection))
        else:
            self.glsl_map = direct_lookup_shade_without_collision.replace(
                '$use_selection', str(use_selection).lower()
            ).replace('$selection', str(selection))


def idx_to_2D(idx, shape):
    """
    From a 1D index generate a 2D index that fits the given shape.

    The 2D index will wrap around line by line and back to the beginning.
    """
    return int((idx // shape[1]) % shape[0]), int(idx % shape[1])


def hash2d_get(key, keys, values, empty_val=0):
    """
    Given a key, retrieve its location in the keys table.
    """
    pos = idx_to_2D(key, keys.shape)
    initial_key = key
    while keys[pos] != initial_key and keys[pos] != empty_val:
        if key - initial_key > keys.size:
            raise KeyError('label does not exist')
        key += 1
        pos = idx_to_2D(key, keys.shape)
    return pos if keys[pos] == initial_key else None


def hash2d_set(key: float, value, keys, values, empty_val=0) -> bool:
    """
    Set a value in the 2d hashmap, wrapping around to avoid collision.
    """
    if key is None or isnan(key):
        return False
    pos = idx_to_2D(key, keys.shape)
    initial_key = key
    collision = False
    while keys[pos] != empty_val:
        collision = True
        if key - initial_key > keys.size:
            raise OverflowError('too many labels')
        key += 1
        pos = idx_to_2D(key, keys.shape)
    keys[pos] = initial_key
    values[pos] = value

    return collision


def _get_shape_from_dict(color_dict):
    size = len(color_dict) * 4
    # I think that hash table size should be at least 4 times
    # bigger than the number of labels to avoid collisions
    for i, prime in enumerate(PRIME_NUM_TABLE[:-1]):
        if prime * prime > size:
            return prime, prime
        if prime * PRIME_NUM_TABLE[i + 1] > size:
            return PRIME_NUM_TABLE[i + 1], prime

    if size > PRIME_NUM_TABLE[-1] * PRIME_NUM_TABLE[-1]:
        raise OverflowError('too many labels')
    return PRIME_NUM_TABLE[-1], PRIME_NUM_TABLE[-1]


def get_shape_from_dict(color_dict):
    global MAX_TEXTURE_SIZE
    if MAX_TEXTURE_SIZE is None:
        MAX_TEXTURE_SIZE = get_max_texture_sizes()[0]

    shape = _get_shape_from_dict(color_dict)

    if MAX_TEXTURE_SIZE is not None and (
        shape[0] > MAX_TEXTURE_SIZE or shape[1] > MAX_TEXTURE_SIZE
    ):
        raise OverflowError(
            f'Too many labels. GPU does not support textures of this size.'
            f' Requested size is {shape[0]}x{shape[1]}, but maximum supported'
            f' size is {MAX_TEXTURE_SIZE}x{MAX_TEXTURE_SIZE}'
        )
    return shape


def build_textures_from_dict(color_dict, empty_val=0, shape=None):
    if len(color_dict) > 2**31 - 2:
        raise OverflowError(
            'Too many labels. Maximum supported number of labels is 2^31-2'
        )

    if shape is None:
        shape = get_shape_from_dict(color_dict)
    keys = np.full(shape, empty_val, dtype=np.float32)
    values = np.zeros(shape + (4,), dtype=np.float32)
    collision = False
    collided = set()
    for key, value in color_dict.items():
        key = np.float32(key)
        if key in collided:
            continue
        collided.add(key)
        collision |= hash2d_set(key, value, keys, values)
    return keys, values, collision


class VispyLabelsLayer(VispyImageLayer):
    def __init__(self, layer, node=None, texture_format='r32f') -> None:
        super().__init__(
            layer,
            node=node,
            texture_format=texture_format,
            layer_node_class=LabelLayerNode,
        )

        self.layer.events.color_mode.connect(self._on_colormap_change)
        self.layer.events.labels_update.connect(self._on_partial_labels_update)
        self.layer.events.selected_label.connect(self._on_colormap_change)
        self.layer.events.show_selected_label.connect(self._on_colormap_change)

    def _on_rendering_change(self):
        # overriding the Image method so we can maintain the same old rendering name
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
            key_texture, val_texture, collision = build_textures_from_dict(
                color_dict
            )
            self.node.cmap = DirectLabelVispyColormap(
                use_selection=colormap.use_selection,
                selection=colormap.selection,
                collision=collision,
            )
            # note that textures have to be transposed here!
            self.node.shared_program['texture2D_keys'] = Texture2D(
                key_texture.T, internalformat='r32f', interpolation='nearest'
            )
            self.node.shared_program['texture2D_values'] = Texture2D(
                val_texture.swapaxes(0, 1),
                internalformat='rgba32f',
                interpolation='nearest',
            )
            self.node.shared_program['LUT_shape'] = key_texture.shape
            self.node.shared_program['color_count'] = len(color_dict) + 1
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


class LabelNode(BaseLabel):  # type: ignore [valid-type,misc]
    def _compute_bounds(self, axis, view):
        if self._data is None:
            return None
        elif axis > 1:  # noqa: RET505
            return (0, 0)
        else:
            return (0, self.size[axis])
