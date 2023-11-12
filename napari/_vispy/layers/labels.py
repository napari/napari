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
from napari._vispy.visuals.labels import LabelNode
from napari._vispy.visuals.volume import Volume as VolumeNode
from napari.utils.colormaps.colormap import minimum_dtype_for_labels

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


auto_lookup_shader = """
uniform sampler2D texture2D_values;

vec4 sample_label_color(float t) {
    // VisPy automatically scales uint8 and uint16 to [0, 1].
    // this line fixes returns values to their original range.
    t = t * $scale;

    if (($use_selection) && ($selection != t)) {
        return vec4(0);
    }
    t = mod(t, $color_map_size);
    return texture2D(
        texture2D_values,
        vec2(0.0, (t + 0.5) / $color_map_size)
    );
}
"""


direct_lookup_shader = """
uniform sampler2D texture2D_values;


vec4 sample_label_color(float t) {
    t = t * $scale;
    if (($use_selection) && ($selection != t)) {
        return vec4(0);
    }
    return texture2D(
        texture2D_values,
        vec2(0.0, (t + 0.5) / $color_map_size)
    );
}

"""


class LabelVispyColormap(VispyColormap):
    def __init__(
        self,
        colors,
        use_selection=False,
        selection=0.0,
        scale=1.0,
    ):
        super().__init__(
            colors=["w", "w"], controls=None, interpolation='zero'
        )
        self.glsl_map = (
            auto_lookup_shader.replace('$color_map_size', str(len(colors)))
            .replace('$use_selection', str(use_selection).lower())
            .replace('$selection', str(selection))
            .replace('$scale', str(scale))
        )


class DirectLabelVispyColormap(VispyColormap):
    def __init__(
        self, use_selection=False, selection=0.0, scale=1.0, color_map_size=255
    ):
        colors = ['w', 'w']  # dummy values, since we use our own machinery
        super().__init__(colors, controls=None, interpolation='zero')
        self.glsl_map = (
            direct_lookup_shader.replace(
                "$use_selection", str(use_selection).lower()
            )
            .replace("$selection", str(selection))
            .replace("$scale", str(scale))
            .replace("$color_map_size", str(color_map_size))
        )


def build_textures_from_dict(
    color_dict: Dict[int, ColorTuple],
) -> np.ndarray:
    data = np.zeros((len(color_dict), 4), dtype=np.float32)
    for key, value in color_dict.items():
        data[key] = value
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

        if mode == 'auto':
            dtype = minimum_dtype_for_labels(self.layer.num_colors + 1)
            if issubclass(dtype.type, np.integer):
                scale = np.iinfo(dtype).max
            else:  # float32 texture
                scale = 1.0
            self.node.cmap = LabelVispyColormap(
                colors=colormap.colors,
                use_selection=colormap.use_selection,
                selection=colormap.selection,
                scale=scale,
            )
            self.node.shared_program['texture2D_values'] = Texture2D(
                colormap.colors.reshape(
                    (colormap.colors.shape[0], 1, 4)
                ).astype(np.float32),
                internalformat='rgba32f',
                interpolation='nearest',
            )

        elif mode == 'direct':
            color_dict = self.layer._direct_colormap.values_mapping_to_minimum_values_set()[
                1
            ]  # TODO: should probably account for non-given labels
            val_texture = build_textures_from_dict(color_dict)

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
            )
            # note that textures have to be transposed here!
            self.node.shared_program['texture2D_values'] = Texture2D(
                val_texture.reshape((val_texture.shape[0], 1, 4)),
                internalformat='rgba32f',
                interpolation='nearest',
            )
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
            else np.array(
                [[0.0]],
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
