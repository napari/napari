import numpy as np
from vispy.gloo import VertexBuffer
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual

_VERTEX_SHADER = """#version 330
in float a_idx;  // int attribvutes not working, so we use a float

out vec4 v_color;
out vec2 v_center;

void main()
{
    int segment = int(a_idx) / 2;
    int endpoint = int(a_idx) % 2;
    int axis = segment / 2;
    float sign = (segment % 2 == 0) ? 1.0 : -1.0;

    vec3 direction =
        axis == 0 ? vec3(1,0,0) :
        axis == 1 ? vec3(0,1,0) :
                    vec3(0,0,1) ;
    direction *= sign;

    // camera-space cursor center
    vec4 center = $visual_to_render(vec4($center,1));

    // point very far along axis
    vec4 far = $visual_to_render(vec4(direction * 1e6, 1));

    // projected direction in screen space
    vec2 center_ndc = (center.xy / center.w);
    vec2 dir_ndc = normalize((far.xy / far.w) - center_ndc);

    float gap = 0.02;  // TODO: make settable
    float extent = 5.0;

    vec2 pos;
    if(endpoint == 0) {
        // inner point
        pos = center_ndc + dir_ndc * gap;
    } else {
        // outer point
        pos = center_ndc + dir_ndc * extent;
    }

    gl_Position = vec4(pos, center.z / center.w, 1.0);

    v_color = $color;
}
"""

_FRAGMENT_SHADER = """#version 330
in vec4 v_color;

void main() {
    gl_FragColor = v_color;
}
"""


class CursorLocatorVisual(Visual):
    def __init__(self):
        super().__init__(vcode=_VERTEX_SHADER, fcode=_FRAGMENT_SHADER)
        self.shared_program['a_idx'] = VertexBuffer(
            np.arange(12, dtype=np.float32)
        )
        self._draw_mode = 'lines'
        self.shared_program.vert['color'] = (1, 0, 0, 1)

    def set_position(self, value: np.ndarray) -> None:
        self.shared_program.vert['center'] = np.array(value, dtype=np.float32)

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


CursorLocator = create_visual_node(CursorLocatorVisual)
