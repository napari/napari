from vispy.scene.visuals import create_visual_node
from vispy.visuals.image import ImageVisual
from vispy.visuals.shaders import Function, FunctionChain

from napari._vispy.visuals.util import TextureMinix

SCALE_R8 = 'float cmap(float v) { return v*255; }'
SCALE_R16 = 'float cmap(float v) { return v*65535; }'


class LabelVisual(TextureMinix, ImageVisual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_color_transform(self):
        funcs = [
            Function(self._func_templates['red_to_luminance']),
            Function(self.cmap.glsl_map),
        ]

        return FunctionChain(
            funcs=funcs,
        )


BaseLabel = create_visual_node(LabelVisual)


class LabelNode(BaseLabel):  # type: ignore [valid-type,misc]
    def _compute_bounds(self, axis, view):
        if self._data is None:
            return None
        elif axis > 1:  # noqa: RET505
            return 0, 0
        else:
            return 0, self.size[axis]
