import numpy as np
from vispy.color import ColorArray
from vispy.color import Colormap as VispyColormap
from vispy.scene.node import Node
from vispy.scene.visuals import create_visual_node
from vispy.visuals.image import ImageVisual
from vispy.visuals.shaders import Function, FunctionChain

from napari._vispy.layers.image import ImageLayerNode, VispyImageLayer
from napari._vispy.visuals.volume import Volume as VolumeNode
from napari.utils.colormaps import low_discrepancy_image
from napari.utils.colormaps.colormap import LabelColormap

# from napari._vispy.layers.base import VispyBaseLayer


def _glsl_label_step(controls=None, colors=None, texture_map_data=None):
    # TODO: Clean up useless shit
    ncolors = len(controls) - 1

    LUT = texture_map_data
    texture_len = texture_map_data.shape[0]
    LUT_tex_idx = np.linspace(0.0, 1.0, texture_len)

    # Replicate indices to colormap texture.
    # The resulting matrix has size of (texture_len,len(controls)).
    # It is used to perform piecewise constant interpolation
    # for each RGBA color component.
    t2 = np.repeat(LUT_tex_idx[:, np.newaxis], len(controls), 1)

    # Perform element-wise comparison to find
    # control points for all LUT colors.
    bn = np.sum(controls.transpose() <= t2, axis=1)

    j = np.clip(bn - 1, 0, ncolors - 1)

    # Copying color data from ColorArray to array-like
    # makes data assignment to LUT faster.
    colors_rgba = ColorArray(colors[:])._rgba
    LUT[:, 0, :] = colors_rgba[j]

    low_disc_plus_cmap = """
    uniform sampler2D texture2D_LUT;

    vec4 low_disc_plus_cmap(float t) {
        float phi_mod = 0.6180339887498948482;  // phi - 1
        float value = 0.0;
        float margin = 1.0 / 256;

        bool use_selection = $use_selection;
        float selection = $selection;

        if (t == 0) {
            return vec4(0.0,0.0,0.0,0.0);
        }

        if ((use_selection) && (selection != t)) {
            return vec4(0.0,0.0,0.0,0.0);
        }

        value = mod((t * phi_mod + $seed), 1.0) * (1 - 2*margin) + margin;

        return texture2D(
            texture2D_LUT,
            vec2(0.0, clamp(value, 0.0, 1.0))
        );
    }
    """
    return low_disc_plus_cmap


direct_colors_in_vispy = """
uniform sampler2D texture2D_keys;
uniform sampler2D texture2D_values;
uniform vec2 shape;

vec4 hash_2d_get(float t) {
    bg_key = 0.;
    vec2 inc = vec2(0., 1.);
    vec2 pos = vec2(mod(t, shape.x), mod(t, shape.y));
    float found = texture2D(
        texture2D_keys,
        pos / shape
    );
    while(found != t || found != bg_key) {
        pos = mod(pos + inc, shape);
        found = texture2D(
            texture2D_keys,
            pos / shape
        );
    }
    return texture2D(
        texture2D_values,
        pos / shape
    );
}

vec4 low_disc_plus_cmap(float t) {
    return hash_2d_get(t);
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
        self.seed = seed
        self.use_selection = use_selection
        self.selection = selection
        self.update_shader()

    def update_shader(self):
        self.glsl_map = (
            _glsl_label_step(
                self._controls, self.colors, self.texture_map_data
            )
            .replace('$seed', str(self.seed))
            .replace('$use_selection', str(self.use_selection).lower())
            .replace('$selection', str(self.selection))
        )
        # Lorenzo says: the VisPy Way(TM)
        # map_func = Function(_glsl_label_step(
        #        self._controls, self.colors, self.texture_map_data
        # ))
        # map_func['seed'] = str(self.seed)
        # self.glsl_map = map_func

    def map(self, x):
        tp = np.where(x == 0, 0, low_discrepancy_image(x, self.seed))
        return super().map(tp)


class LabelVispyDirectColormap(VispyColormap):
    def __init__(
        self,
        color_dict,
        use_selection=False,
        selection=0.0,
    ):
        super().__init__(
            np.zeros((1, 4)), np.array([0.0, 1.0]), interpolation='zero'
        )
        self.use_selection = use_selection
        self.selection = selection
        self.update_shader()

    def update_shader(self):
        self.glsl_map = (
            _glsl_label_step(
                self._controls, self.colors, self.texture_map_data
            )
            .replace('$seed', str(self.seed))
            .replace('$use_selection', str(self.use_selection).lower())
            .replace('$selection', str(self.selection))
        )
        # Lorenzo says: the VisPy Way(TM)
        # map_func = Function(_glsl_label_step(
        #        self._controls, self.colors, self.texture_map_data
        # ))
        # map_func['seed'] = str(self.seed)
        # self.glsl_map = map_func

    def map(self, x):
        tp = np.where(x == 0, 0, low_discrepancy_image(x, self.seed))
        return super().map(tp)


def idx_to_2D(idx, shape):
    return (idx % shape[0], idx % shape[1])


def hash2d_get(key, keys, values, empty_val=0.0):
    pos = idx_to_2D(key, keys.shape)
    while keys[pos] != key and keys[pos] != empty_val:
        pos = (pos[0], (pos[1] + 1) % keys.shape[1])
    if keys[pos] == key:
        return pos, values[pos]
    else:
        return None, None


def hash2d_set(key, value, keys, values, empty_val=0.0):
    pos = (key % keys.shape[0], key % keys.shape[1])
    while keys[pos] != empty_val:
        pos = (pos[0], (pos[1] + 1) % keys.shape[1])
    keys[pos] = key
    values[pos] = value


def build_textures_from_dict(color_dict):
    shape = (1019, 1021)
    keys = np.zeros(shape, dtype=np.float32)
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

        if isinstance(colormap, LabelColormap) and mode == 'auto':
            self.node.cmap = LabelVispyColormap(
                colors=colormap.colors,
                controls=colormap.controls,
                seed=colormap.seed,
                use_selection=colormap.use_selection,
                selection=colormap.selection,
            )
        elif mode == 'direct':
            color_dict = self.layer.colors
            key_texture, val_texture = build_textures_from_dict(color_dict)
            self.node.shared_program['texture2D_keys'] = key_texture
            self.node.shared_program['texture2D_colors'] = val_texture
            self.node.cmap = LabelVispyColormap()
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
