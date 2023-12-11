from itertools import product
from math import ceil, isnan, log2, sqrt
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

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
from napari.utils._dtype import vispy_texture_dtype
from napari.utils.colormaps.colormap import (
    LabelColormap,
)

if TYPE_CHECKING:
    from napari.layers import Labels


# We use table sizes that are prime numbers near powers of 2.
# For each power of 2, we keep three candidate sizes. This allows us to
# maximize the chances of finding a collision-free table for a given set of
# keys (which we typically know at build time).
PRIME_NUM_TABLE = [
    [37, 31, 29],
    [61, 59, 53],
    [127, 113, 109],
    [251, 241, 239],
    [509, 503, 499],
    [1021, 1019, 1013],
    [2039, 2029, 2027],
    [4093, 4091, 4079],
    [8191, 8179, 8171],
    [16381, 16369, 16363],
    [32749, 32719, 32717],
    [65521, 65519, 65497],
]

START_TWO_POWER = 5

MAX_LOAD_FACTOR = 0.25

MAX_TEXTURE_SIZE = None

ColorTuple = Tuple[float, float, float, float]


EMPTY_VAL = -1.0

_UNSET = object()


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
uniform sampler2D texture2D_keys;
uniform sampler2D texture2D_values;
uniform vec2 LUT_shape;
uniform int color_count;


vec4 sample_label_color(float t) {
    if (($use_selection) && ($selection != t)) {
        return vec4(0);
    }

    float empty = $EMPTY_VAL;
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

    vec4 color = vec4($default_color);
    if (abs(found - empty) > 1e-8) {
        color = texture2D(
            texture2D_values,
            pos_tex
        );
    }
    return color;
}

"""


class LabelVispyColormap(VispyColormap):
    def __init__(
        self,
        colormap: LabelColormap,
        view_dtype: np.dtype,
        raw_dtype: np.dtype,
    ):
        super().__init__(
            colors=["w", "w"], controls=None, interpolation='zero'
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
                f"Cannot use dtype {view_dtype} with LabelVispyColormap"
            )

        selection = colormap.selection_as_minimum_dtype(raw_dtype)

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
        collision=True,
        default_color=(0, 0, 0, 0),
        empty_value=EMPTY_VAL,
    ):
        colors = ['w', 'w']  # dummy values, since we use our own machinery
        super().__init__(colors, controls=None, interpolation='zero')
        self.glsl_map = (
            direct_lookup_shader.replace(
                "$use_selection", str(use_selection).lower()
            )
            .replace("$selection", str(selection))
            .replace("$collision", str(collision).lower())
            .replace("$default_color", ", ".join(map(str, default_color)))
            .replace("$EMPTY_VAL", str(empty_value))
        )


def idx_to_2d(idx, shape):
    """
    From a 1D index generate a 2D index that fits the given shape.

    The 2D index will wrap around line by line and back to the beginning.
    """
    return int((idx // shape[1]) % shape[0]), int(idx % shape[1])


def hash2d_get(key, keys, empty_val=EMPTY_VAL):
    """
    Given a key, retrieve its location in the keys table.
    """
    pos = idx_to_2d(key, keys.shape)
    initial_key = key
    while keys[pos] != initial_key and keys[pos] != empty_val:
        if key - initial_key > keys.size:
            raise KeyError('label does not exist')
        key += 1
        pos = idx_to_2d(key, keys.shape)
    return pos if keys[pos] == initial_key else None


def hash2d_set(
    key: Union[float, np.floating],
    value: ColorTuple,
    keys: np.ndarray,
    values: np.ndarray,
    empty_val=EMPTY_VAL,
) -> bool:
    """
    Set a value in the 2d hashmap, wrapping around to avoid collision.
    """
    if key is None or isnan(key):
        return False
    pos = idx_to_2d(key, keys.shape)
    initial_key = key
    collision = False
    while keys[pos] != empty_val:
        collision = True
        if key - initial_key > keys.size:
            raise OverflowError('too many labels')
        key += 1
        pos = idx_to_2d(key, keys.shape)
    keys[pos] = initial_key
    values[pos] = value

    return collision


def _get_shape_from_keys(
    keys: np.ndarray, first_dim_index: int, second_dim_index: int
) -> Optional[Tuple[int, int]]:
    """Get the smallest hashmap size without collisions, if any.

    This function uses precomputed prime numbers from PRIME_NUM_TABLE.

    For each index, it gets a list of prime numbers close to
    ``2**(index + START_TWO_POWER)`` (where ``START_TWO_POWER=5``), that is,
    the smallest table is close to ``32 * 32``.

    The function then iterates over all combinations of prime numbers from the
    lists and checks for a combination that has no collisions for the
    given keys, returning that combination.

    If no combination can be found, returns None.

    Although keys that collide for all table combinations are rare, they are
    possible: see ``test_collide_keys`` and ``test_collide_keys2``.

    Parameters
    ----------
    keys : np.ndarray
        array of keys to be inserted into the hashmap,
        used for collision detection
    first_dim_index : int
        index for first dimension of PRIME_NUM_TABLE
    second_dim_index : int
        index for second dimension of PRIME_NUM_TABLE

    Returns
    shp : 2-tuple of int, optional
        If a table shape can be found that has no collisions for the given
        keys, return that shape. Otherwise, return None.
    """
    for fst_size, snd_size in product(
        PRIME_NUM_TABLE[first_dim_index],
        PRIME_NUM_TABLE[second_dim_index],
    ):
        fst_crd = (keys // snd_size) % fst_size
        snd_crd = keys % snd_size

        collision_set = set(zip(fst_crd, snd_crd))
        if len(collision_set) == len(set(keys)):
            return fst_size, snd_size
    return None


def _get_shape_from_dict(
    color_dict: Dict[float, Tuple[float, float, float, float]]
) -> Tuple[int, int]:
    """Compute the shape of a 2D hashmap based on the keys in `color_dict`.

    This function finds indices for the first and second dimensions of a
    table in PRIME_NUM_TABLE based on a target load factor of 0.125-0.25,
    then calls `_get_shape_from_keys` based on those indices.

    This is quite a low load-factor, but, ultimately, the hash table
    textures are tiny compared to most datasets, so we choose these
    factors to minimize the chance of collisions and trade a bit of GPU
    memory for speed.
    """
    keys = np.array([x for x in color_dict if x is not None], dtype=np.int64)

    size = len(keys) / MAX_LOAD_FACTOR
    size_sqrt = sqrt(size)
    size_log2 = log2(size_sqrt)
    max_idx = len(PRIME_NUM_TABLE) - 1
    max_size = PRIME_NUM_TABLE[max_idx][0] ** 2
    fst_dim = min(max(int(ceil(size_log2)) - START_TWO_POWER, 0), max_idx)
    snd_dim = min(max(int(round(size_log2, 0)) - START_TWO_POWER, 0), max_idx)

    if len(keys) > max_size:
        raise MemoryError(
            f'Too many labels: napari supports at most {max_size} labels, '
            f'got {len(keys)}.'
        )

    shp = _get_shape_from_keys(keys, fst_dim, snd_dim)
    if shp is None and snd_dim < max_idx:
        # if we still have room to grow, try the next size up to get a
        # collision-free table
        shp = _get_shape_from_keys(keys, fst_dim, snd_dim + 1)
    if shp is None:
        # at this point, if there's still collisions, we give up and return
        # the largest possible table given these indices and the target load
        # factor.
        # (To see a set of keys that cause collision,
        # and land on this branch, see test_collide_keys2.)
        shp = PRIME_NUM_TABLE[fst_dim][0], PRIME_NUM_TABLE[snd_dim][0]
    return shp


def get_shape_from_dict(color_dict):
    global MAX_TEXTURE_SIZE
    if MAX_TEXTURE_SIZE is None:
        MAX_TEXTURE_SIZE = get_max_texture_sizes()[0]

    shape = _get_shape_from_dict(color_dict)

    if MAX_TEXTURE_SIZE is not None and (
        shape[0] > MAX_TEXTURE_SIZE or shape[1] > MAX_TEXTURE_SIZE
    ):
        raise MemoryError(
            f'Too many labels. GPU does not support textures of this size.'
            f' Requested size is {shape[0]}x{shape[1]}, but maximum supported'
            f' size is {MAX_TEXTURE_SIZE}x{MAX_TEXTURE_SIZE}'
        )
    return shape


def _get_empty_val_from_dict(color_dict):
    empty_val = EMPTY_VAL
    while empty_val in color_dict:
        empty_val -= 1
    return empty_val


def build_textures_from_dict(
    color_dict: Dict[Optional[float], ColorTuple],
    empty_val=_UNSET,
    shape=None,
    use_selection=False,
    selection=0.0,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    This function construct hash table for fast lookup of colors.
    It uses pair of textures.
    First texture is a table of keys, used to determine position,
    second is a table of values.

    The procedure of selection table and collision table is
    implemented in hash2d_get function.

    Parameters
    ----------
    color_dict : Dict[float, Tuple[float, float, float, float]]
        Dictionary from labels to colors
    empty_val : float
        Value to use for empty cells in the hash table
    shape : Optional[Tuple[int, int]]
        Shape of the hash table.
        If None, it is calculated from the number of
        labels using _get_shape_from_dict
    use_selection : bool
        If True, only the selected label is shown.
        The generated colormap is single-color of size (1, 1)
    selection : float
        used only if use_selection is True.
        Determines the selected label.

    Returns
    -------
    keys: np.ndarray
        Texture of keys for the hash table
    values: np.ndarray
        Texture of values for the hash table
    collision: bool
        True if there are collisions in the hash table
    """
    if use_selection:
        keys = np.full((1, 1), selection, dtype=vispy_texture_dtype)
        values = np.zeros((1, 1, 4), dtype=vispy_texture_dtype)
        values[0, 0] = color_dict.get(selection, color_dict[None])
        return keys, values, False

    if empty_val is _UNSET:
        empty_val = _get_empty_val_from_dict(color_dict)

    if len(color_dict) > 2**31 - 2:
        raise MemoryError(
            f'Too many labels ({len(color_dict)}). Maximum supported number of labels is 2^31-2'
        )

    if shape is None:
        shape = get_shape_from_dict(color_dict)

    if len(color_dict) > shape[0] * shape[1]:
        raise MemoryError(
            f'Too many labels ({len(color_dict)}). Maximum supported number of labels for the given shape is {shape[0] * shape[1]}'
        )

    keys = np.full(shape, empty_val, dtype=vispy_texture_dtype)
    values = np.zeros(shape + (4,), dtype=vispy_texture_dtype)
    visited = set()
    collision = False
    for key, value in color_dict.items():
        key_ = vispy_texture_dtype(key)
        if key_ in visited:
            # input int keys are unique but can map to the same float.
            # if so, we ignore all but the first appearance.
            continue
        visited.add(key_)
        collision |= hash2d_set(key_, value, keys, values, empty_val=empty_val)

    return keys, values, collision


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

        if mode == 'auto':
            view_dtype = self.layer._slice.image.view.dtype
            raw_dtype = self.layer._slice.image.raw.dtype
            if view_dtype != raw_dtype:
                # If the view dtype is different from the raw dtype, it is possible
                # that background pixels are not the same value as the `background_value`.
                # For example, if raw_dtype is int8 and background_value is `-1`
                # then in view dtype uint8, the background pixels will be 255
                # For data types with more than 16 bits we always cast
                # to uint8 or uint16 and background_value is always 0 in a view array.
                # The LabelColormap is EventedModel, so we need to make
                # a copy instead of temporary overwrite the background_value
                colormap = LabelColormap(**colormap.dict())
                colormap.background_value = (
                    colormap.background_as_minimum_dtype(raw_dtype)
                )
            if view_dtype == np.uint8:
                color_texture = colormap._uint8_colors.reshape(256, -1, 4)
            else:
                color_texture = colormap._uint16_colors.reshape(256, -1, 4)
            self.node.cmap = LabelVispyColormap(
                colormap, view_dtype=view_dtype, raw_dtype=raw_dtype
            )
            self.node.shared_program['texture2D_values'] = Texture2D(
                color_texture,
                internalformat='rgba32f',
                interpolation='nearest',
            )
            self.texture_data = color_texture

        elif mode == 'direct':
            color_dict = (
                self.layer.color
            )  # TODO: should probably account for non-given labels
            key_texture, val_texture, collision = build_textures_from_dict(
                color_dict,
                use_selection=colormap.use_selection,
                selection=float(colormap.selection),
            )

            self.node.cmap = DirectLabelVispyColormap(
                use_selection=colormap.use_selection,
                selection=float(colormap.selection),
                collision=collision,
                default_color=colormap.default_color,
                empty_value=_get_empty_val_from_dict(color_dict),
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
            self.node.shared_program['color_count'] = len(color_dict)
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
