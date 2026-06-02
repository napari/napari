import numpy as np
from vispy.gloo import VertexBuffer
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual

_VERTEX_SHADER = """#version 330
in float a_idx;  // int attribvutes not working, so we use a float

out vec2 v_center;

void main()
{
    // depending on the index of this vertex, we decide which case we're in
    int segment = int(a_idx) / 2;
    int endpoint = int(a_idx) % 2;
    int axis = segment / 2;
    int sign = (segment % 2 == 0) ? 1 : -1;

    vec3 direction =
        axis == 0 ? vec3(sign, 0, 0) :
        axis == 1 ? vec3(0, sign, 0) :
                    vec3(0, 0, sign) ;

    vec2 view_dir = $visual_to_render(vec4(direction, 0)).xy;

    if (length(view_dir) < 1e-5)
    {
        // basically axis-aligned view direction, so we drop this axis
        // by putting it outside of the clip range
        gl_Position = vec4(-2, -2, 0, 1);
        return;
    }

    // camera-space cursor center
    vec4 center = $visual_to_render(vec4($center, 1));

    // projected direction in screen space
    vec2 dir_ndc = normalize(view_dir);
    vec2 center_ndc = (center.xy / center.w);

    float extent = 5.0;  // should be enough to always go out of screen

    vec2 pos;
    if (endpoint == 0) {
        // inner point
        pos = center_ndc + dir_ndc * $gap;
    } else {
        // outer point
        pos = center_ndc + dir_ndc * extent;
    }

    gl_Position = vec4(pos, center.z / center.w, 1.0);
}
"""

_FRAGMENT_SHADER = """#version 330
void main() {
    gl_FragColor = $color;
}
"""


class CrosshairVisual(Visual):
    def __init__(self):
        super().__init__(vcode=_VERTEX_SHADER, fcode=_FRAGMENT_SHADER)
        self.shared_program['a_idx'] = VertexBuffer(
            np.arange(12, dtype=np.float32)
        )
        self._draw_mode = 'lines'
        self.position = (0, 0, 0)
        self.color = (1, 1, 1, 1)
        self.gap = 0.05

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        self._position = np.array(value, dtype=np.float32)
        self.shared_program.vert['center'] = self._position
        self.update()

    @property
    def color(self) -> np.ndarray:
        return self._color

    @color.setter
    def color(self, value: np.ndarray) -> None:
        self._color = np.array(value, dtype=np.float32)
        self.shared_program.frag['color'] = self._color
        self.update()

    @property
    def gap(self) -> float:
        return self._gap

    @gap.setter
    def gap(self, value: float) -> None:
        self._gap = float(value)
        self.shared_program.vert['gap'] = self._gap
        self.update()

    def _prepare_transforms(self, view=None):
        view.view_program.vert['visual_to_render'] = (
            view.transforms.get_transform('visual', 'render')
        )

    def _prepare_draw(self, view=None):
        """This method is called immediately before each draw.

        The *view* argument indicates which view is about to be drawn.
        """

        self.update_gl_state(line_smooth=1)
        px_scale = self.transforms.pixel_scale
        width = px_scale * 1
        self.update_gl_state(line_width=max(width, 1.0))


Crosshair = create_visual_node(CrosshairVisual)
