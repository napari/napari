import numpy as np
import vispy
from vispy.color import ColorArray
from vispy.gloo import VertexBuffer
from vispy.scene.visuals import create_visual_node
from vispy.visuals.visual import Visual

vispy.use(gl='gl+')


vert = """
uniform float u_px_scale;
uniform bool u_scaling;
uniform float u_width;

attribute vec3 a_position;
attribute vec4 a_color;

varying vec4 v_color;
varying float v_width;

float big_float = 1e10; // prevents numerical imprecision

void main (void) {
    v_color = a_color;

    vec4 pos = vec4(a_position, 1);
    vec4 fb_pos = $visual_to_framebuffer(pos);
    gl_Position = $framebuffer_to_render(fb_pos);

    if (u_scaling == true) {
        // calculate point size from visual to framebuffer coords to determine size
        vec4 x = $framebuffer_to_visual(fb_pos + vec4(big_float, 0, 0, 0));
        x = (x - pos);
        vec4 size_vec = $visual_to_framebuffer(pos + normalize(x) * u_width);
        v_width = (size_vec.x - fb_pos.x) / 2;
    }
    else {
        v_width = u_width * u_px_scale;
    }
}
"""

geom = """
#version 150
layout (lines) in;
layout (triangle_strip, max_vertices=4) out;

in vec4 v_color[];
in float v_width[];

out vec4 v_color_out;

void main(void) {
    // start and end position of the cylinder
    vec4 start = gl_in[0].gl_Position;
    vec4 end = gl_in[1].gl_Position;

    // calculcations need to happen in framebuffer coords or clipping messes up
    vec4 start_fb = $render_to_framebuffer(start);
    vec4 end_fb = $render_to_framebuffer(end);

    // find the vector perpendicular to the cylinder direction projected on the screen
    vec4 direction = end_fb / end_fb.w - start_fb / start_fb.w;
    vec4 perp_screen = normalize(vec4(direction.y, -direction.x, 0, 0));

    vec4 shift_start = $framebuffer_to_render(perp_screen * v_width[0]);
    gl_Position = start + shift_start;
    v_color_out = v_color[0];
    EmitVertex();

    gl_Position = start - shift_start;
    v_color_out = v_color[0];
    EmitVertex();

    vec4 shift_end = $framebuffer_to_render(perp_screen * v_width[1]);
    gl_Position = end + shift_end;
    v_color_out = v_color[1];
    EmitVertex();

    gl_Position = end - shift_end;
    v_color_out = v_color[1];
    EmitVertex();

    EndPrimitive();
}
"""

frag = """
varying vec4 v_color_out;

void main()
{
    gl_FragColor = v_color_out;
}
"""


class LineVisual(Visual):
    def __init__(self, width=5, scaling=True, **kwargs):
        self._vbo = VertexBuffer()
        self._data = None

        Visual.__init__(self, vcode=vert, gcode=geom, fcode=frag)

        self._draw_mode = 'lines'

        if len(kwargs) > 0:
            self.set_data(**kwargs)

        self.width = width
        self.scaling = scaling

        self.freeze()

    def set_data(self, pos=None, color='white'):
        color = ColorArray(color).rgba
        if len(color) == 1:
            color = color[0]

        if pos is not None:
            assert isinstance(pos, np.ndarray)
            assert pos.ndim == 2
            assert pos.shape[1] in (2, 3)

            data = np.zeros(
                len(pos),
                dtype=[
                    ('a_position', np.float32, 3),
                    ('a_color', np.float32, 4),
                ],
            )
            data['a_color'] = color
            data['a_position'][:, : pos.shape[1]] = pos
            self._data = data
            self._vbo.set_data(data)
            self.shared_program.bind(self._vbo)

        self.update()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self.shared_program['u_width'] = value
        self._width = value
        self.update()

    @property
    def scaling(self):
        """
        If set to True, width scales when rezooming.
        """
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        value = bool(value)
        self.shared_program['u_scaling'] = value
        self._scaling = value
        self.update()

    def _prepare_transforms(self, view):
        view.view_program.vert['visual_to_framebuffer'] = view.get_transform(
            'visual', 'framebuffer'
        )
        view.view_program.vert['framebuffer_to_render'] = view.get_transform(
            'framebuffer', 'render'
        )
        view.view_program.vert['framebuffer_to_visual'] = view.get_transform(
            'framebuffer', 'visual'
        )
        view.view_program.geom['render_to_framebuffer'] = view.get_transform(
            'render', 'framebuffer'
        )
        view.view_program.geom['framebuffer_to_render'] = view.get_transform(
            'framebuffer', 'render'
        )

    def _prepare_draw(self, view):
        if self._data is None:
            return False
        view.view_program['u_px_scale'] = view.transforms.pixel_scale
        view.view_program['u_scaling'] = self.scaling
        return True

    def _compute_bounds(self, axis, view):
        if self._data is None:
            return None
        pos = self._data['a_position']
        if pos is None:
            return None
        if pos.shape[1] > axis:
            return (pos[:, axis].min(), pos[:, axis].max())
        return (0, 0)


Line = create_visual_node(LineVisual)
