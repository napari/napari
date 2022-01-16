import numpy as np
from vispy.color import ColorArray
from vispy.color import Colormap as VispyColormap

from ...utils.colormaps import low_discrepancy_image
from .image import VispyImageLayer


def _glsl_label_step(controls=None, colors=None, texture_map_data=None):
    assert (controls[0], controls[-1]) == (0.0, 1.0)
    ncolors = len(controls) - 1
    assert ncolors >= 2
    assert texture_map_data is not None

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

    low_disc = """
    float low_disc(float t) {
        float phi_mod = 0.6180339887498948482;  // phi - 1
        float value = 0.0;

        if (t == 0) {
            return t;
        }

        t = (t * phi_mod + $seed);
        t = mod(value, 1.0);

        return t;
    }
    """
    s2 = "uniform sampler2D texture2D_LUT;"
    s = "{\n return texture2D(texture2D_LUT, \
           vec2(0.0, clamp(low_disc(t), 0.0, 1.0)));\n} "

    return f"{low_disc}\n{s2}\nvec4 colormap(float t) {{\n{s}\n}}"


class LabelColormap(VispyColormap):
    def __init__(self, colors, controls=None, seed=0.5):
        super().__init__(colors, controls, interpolation='zero')
        self.seed = seed
        self.update_shader()

    def update_shader(self):
        self.glsl_map = _glsl_label_step(
            self._controls, self.colors, self.texture_map_data
        ).replace('$seed', str(self.seed))

        print(self.glsl_map)

    def map(self, x):
        tp = np.where(x == 0, 0, low_discrepancy_image(x, self.seed))
        return super().map(tp)


class VispyLabelsLayer(VispyImageLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, texture_format=None, **kwargs)

    def _on_colormap_change(self, event=None):
        # self.layer.colormap is a labels_colormap, which is an evented model
        # from napari.utils.colormaps.Colormap (or similar). If we use it
        # in our constructor, we have access to the texture data we need
        colormap = self.layer.colormap
        self.node.cmap = LabelColormap(
            colors=colormap.colors,
            controls=colormap.controls,
            seed=colormap.seed,
        )
