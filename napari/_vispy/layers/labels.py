import numpy as np
from vispy.color import ColorArray
from vispy.color import Colormap as VispyColormap
from vispy.scene.node import Node
from vispy.scene.visuals import create_visual_node
from vispy.visuals.image import ImageVisual
from vispy.visuals.shaders import Function, FunctionChain

from ...utils.colormaps import low_discrepancy_image
from ..visuals.volume import Volume as VolumeNode
from .image import ImageLayerNode, VispyImageLayer

# from napari._vispy.layers.base import VispyBaseLayer


def _glsl_label_step(controls=None, colors=None, texture_map_data=None):
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

        if (t == 0) {
            return vec4(0.0,0.0,0.0,0.0);
        }

        value = mod((t * phi_mod + $seed), 1.0);

        return texture2D(
            texture2D_LUT,
            vec2(0.0, clamp(value, 0.0, 1.0))
        );
    }
    """
    return low_disc_plus_cmap


class LabelVispyColormap(VispyColormap):
    def __init__(self, colors, controls=None, seed=0.5):
        super().__init__(colors, controls, interpolation='zero')
        self.seed = seed
        self.update_shader()

    def update_shader(self):
        self.glsl_map = _glsl_label_step(
            self._controls, self.colors, self.texture_map_data
        ).replace('$seed', str(self.seed))

    def map(self, x):
        tp = np.where(x == 0, 0, low_discrepancy_image(x, self.seed))
        return super().map(tp)


class VispyLabelsLayer(VispyImageLayer):
    def __init__(self, layer, node=None, texture_format='r32f'):
        super().__init__(
            layer,
            node=node,
            texture_format=texture_format,
            layer_node_class=LabelLayerNode,
        )

    def _on_colormap_change(self, event=None):
        # self.layer.colormap is a labels_colormap, which is an evented model
        # from napari.utils.colormaps.Colormap (or similar). If we use it
        # in our constructor, we have access to the texture data we need
        colormap = self.layer.colormap
        self.node.cmap = LabelVispyColormap(
            colors=colormap.colors,
            controls=colormap.controls,
            seed=colormap.seed,
        )


class LabelVisual(ImageVisual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_color_transform(self):
        if self._data.ndim == 2 or self._data.shape[2] == 1:
            fun = FunctionChain(
                None,
                [
                    Function(self._func_templates['red_to_luminance']),
                    Function(self.cmap.glsl_map),
                ],
            )
        else:
            # RGB/A image data (no colormap)
            raise Exception("TODO: RGB/A lookups")
            fclim = Function(self._func_templates['clim'])
            fgamma = Function(self._func_templates['gamma'])
            fun = FunctionChain(
                None,
                [
                    Function(self._func_templates['null_color_transform']),
                    fclim,
                    fgamma,
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

        # TODO
        self._volume_node = VolumeNode(
            np.zeros((1, 1, 1), dtype=np.float32),
            clim=[0, 1],
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
