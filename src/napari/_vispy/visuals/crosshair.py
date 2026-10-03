from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vispy.gloo import VertexBuffer
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual

if TYPE_CHECKING:
    from vispy.visuals.visual import VisualView

_VERTEX_SHADER = """#version 330

// this shader has no vertices; instead, it uses indices to distinguish
// between which case we're in (i.e: which axis and which extremity).
// `axis` will be 0, 1 or 2, and `side` either -1 or 1. Those are used to
// output the segment extremities far past the screen so the lines look
// infinite. The central gap is then discarded in the fragment shader.

in float a_idx;  // int attributes not working, so we use a float

out vec2 v_center;

void main()
{
    int axis = int(a_idx) / 2;
    float side = (int(a_idx) % 2 == 0) ? -1.0 : 1.0;

    vec3 direction =
        axis == 0 ? vec3(side, 0, 0) :
        axis == 1 ? vec3(0, side, 0) :
                    vec3(0, 0, side);

    // axis vector on screen (non-normalized)
    vec2 axis_dir = $visual_to_render(vec4(direction, 0)).xy;

    if (length(axis_dir) < 1e-5)
    {
        // very small, so we're basically looking straight down this axis; we drop
        // it by putting it outside of the clip range (cannot discard in vertex)
        gl_Position = vec4(-2, -2, 0, 1);
        return;
    }

    // pass the center position in ndc coordinates to the fragment shader
    vec4 center = $visual_to_render(vec4($center, 1));
    vec2 center_ndc = (center.xy / center.w);
    v_center = center_ndc;

    float extent = 5.0;  // should be enough to always go out of screen

    // position of this vertex (the extremity), far past the screen edge
    // in the direction of the axis, starting from the center
    vec2 pos = center_ndc + normalize(axis_dir) * extent;

    gl_Position = vec4(pos, center.z / center.w, 1.0);
}
"""

_FRAGMENT_SHADER = """#version 330
in vec2 v_center;

void main() {
    // gl_FragCoord is in physical pixels; $canvas_size is logical pixels
    vec2 screen_pos = gl_FragCoord.xy / $pixel_ratio;
    vec2 center_px = (v_center * 0.5 + 0.5) * $canvas_size;
    float dist_from_center = length(screen_pos - center_px);

    // discard fragments inside the circular gap
    if (dist_from_center < $gap / 2.0) {
        discard;
    }
    gl_FragColor = $color;
}
"""


class CrosshairVisual(Visual):
    """Crosshair visual with a central gap.

    Displays an "infinite" 3D crosshair around a central point, with
    a circular gap around the center defined in screen pixels.
    """

    def __init__(self) -> None:
        super().__init__(vcode=_VERTEX_SHADER, fcode=_FRAGMENT_SHADER)
        self.shared_program['a_idx'] = VertexBuffer(
            np.arange(6, dtype=np.float32)
        )
        self._draw_mode = 'lines'
        self.position = np.array((0, 0, 0))
        self.color = np.array((1, 1, 1, 1))
        self.gap = 20

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
        self.shared_program.frag['gap'] = self._gap
        self.update()

    def _prepare_transforms(self, view: VisualView | None = None) -> None:
        if view is not None:
            view.view_program.vert['visual_to_render'] = (
                view.transforms.get_transform('visual', 'render')
            )

    def _prepare_draw(self, view: VisualView | None = None) -> None:
        """This method is called immediately before each draw.

        The *view* argument indicates which view is about to be drawn.
        """

        self.update_gl_state(line_smooth=1)
        px_scale = self.transforms.pixel_scale
        width = px_scale * 1
        self.update_gl_state(line_width=max(width, 1.0))

        if view is not None:
            self.shared_program.frag['canvas_size'] = view.canvas.size
            self.shared_program.frag['pixel_ratio'] = px_scale


Crosshair = create_visual_node(CrosshairVisual)
