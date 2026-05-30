"""Instanced rendering for vectors using vispy 0.16+."""

import numpy as np
import vispy
from vispy.gloo import IndexBuffer, VertexBuffer
from vispy.scene.visuals import create_visual_node
from vispy.visuals.visual import Visual

from napari._vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin

vispy.use(gl='gl+')  # Required for instanced rendering

# Vertex shader for instanced vector rendering
vert = r"""
uniform float u_px_scale;
uniform float u_width;
uniform float u_data_scale;  // Data-to-screen scale factor
uniform float u_head_width_ratio;  // Arrow head width / shaft width
uniform float u_head_length_ratio;  // Arrow head length / total length

// Base geometry vertices (per-vertex, reused for all instances)
attribute vec2 a_template_pos;  // x: along vector [0-1], y: perpendicular (sign=direction, mag encodes type)

// Instance attributes (per-vector)
attribute vec3 a_start;
attribute vec3 a_end;
attribute vec4 a_color;

varying vec4 v_color;

void main (void) {
    v_color = a_color;

    // Transform start and end positions to framebuffer space
    vec4 start_fb = $visual_to_framebuffer(vec4(a_start, 1.0));
    vec4 end_fb = $visual_to_framebuffer(vec4(a_end, 1.0));

    // Calculate vector direction and perpendicular in screen space for billboarding
    vec4 direction_fb = end_fb / end_fb.w - start_fb / start_fb.w;
    vec4 perp_fb = normalize(vec4(direction_fb.y, -direction_fb.x, 0, 0));

    // Interpret template position:
    // x: position along vector, shaft=[0,0.5], head=[0.5,1.0]
    // y: perpendicular offset (half-width):
    //    - Shaft: ±0.25
    //    - Head base: ±0.75 (3x shaft, giving default 3:1 ratio)
    //    - Head tip: 0.0
    //
    // Template layout (seven vertices total):
    //  (0, 0.25)
    //  A---------B (0.5, 0.25)    E (0.5, 0.75)
    //  |         |                 \
    //  |         |                  \
    //  |         |                   G (1.0, 0.0)
    //  |         |                  /
    //  D---------C (0.5, -0.25)    /
    // (0, -0.25)                  F (0.5, -0.75)

    // Remap x coordinate based on head/shaft ratio
    // Shaft [0, 0.5] → [0, shaft_end], Head [0.5, 1.0] → [shaft_end, 1.0]
    float shaft_end = 1.0 - u_head_length_ratio;
    float remapped_x;

    if (a_template_pos.x <= 0.5) {
        // Shaft: remap [0, 0.5] → [0, shaft_end]
        remapped_x = a_template_pos.x * 2.0 * shaft_end;
    } else {
        // Head: remap [0.5, 1.0] → [shaft_end, 1.0]
        remapped_x = shaft_end + (a_template_pos.x - 0.5) * 2.0 * (1.0 - shaft_end);
    }

    // Interpolate position along the vector in framebuffer space
    vec4 pos_fb = mix(start_fb, end_fb, remapped_x);

    // Calculate width in data/world space (scales with zoom)
    // Use precomputed direction-independent scale factor
    float width_fb = u_width * 0.5 * u_data_scale;

    // Scale perpendicular offset based on vertex type
    // Shaft (|y| ≤ 0.5): scale = 2.0 (so 0.25 becomes 0.5 = half-width)
    // Head (|y| > 0.5): scale = 2.0 * head_width_ratio / 3.0 (so 0.75 becomes head half-width)
    float is_head = step(0.5, abs(a_template_pos.y));
    float y_scale = mix(2.0, 2.0 * u_head_width_ratio / 3.0, is_head);

    vec4 offset = perp_fb * a_template_pos.y * width_fb * y_scale;
    pos_fb += offset;

    gl_Position = $framebuffer_to_render(pos_fb);
}
"""

frag = """
varying vec4 v_color;

void main()
{
    gl_FragColor = v_color;
}
"""


class VectorsVisual(ClippingPlanesMixin, Visual):
    """Instanced rendering visual for vectors.

    Supports three vector styles:
    - 'line': Rectangular line segments
    - 'triangle': Triangular arrows
    - 'arrow': Arrows with distinct shaft and head
    """

    def __init__(self, **kwargs):
        self._data = None
        self._n_instances = 0
        self._width = 1.0
        self._vector_style = 'line'

        Visual.__init__(self, vcode=vert, fcode=frag)

        self._draw_mode = 'triangles'

        # Now create buffers after Visual.__init__
        self._instance_vbo = VertexBuffer()
        self._template_vbo = VertexBuffer()
        self._index_buffer = IndexBuffer()

        self._setup_template_vertices()

        if len(kwargs) > 0:
            self.set_data(**kwargs)

    def _setup_template_vertices(self):
        """Setup unified template geometry for all vector styles.

        All styles (line, triangle, arrow) use the same arrow template.
        The actual appearance is controlled by uniforms:
        - line: u_head_length_ratio = 0.0 (all shaft, no head)
        - triangle: u_head_length_ratio = 1.0 (all head, no shaft)
        - arrow: u_head_length_ratio ∈ (0, 1) (shaft + head)

        Template vertex format (vec2):
        - x: position along vector [0, 0.5] for shaft, [0.5, 1.0] for head
        - y: perpendicular offset (half-width):
          * Shaft vertices: y = ±0.25 (will be scaled by edge_width)
          * Head base vertices: y = ±0.75 (gives default 3:1 width ratio)
          * Head tip: y = 0.0 (stays on centerline)
        """
        # Create structured array with just vec2 position
        vertex_dtype = [('a_template_pos', np.float32, 2)]
        vertices = np.zeros(7, dtype=vertex_dtype)

        # Shaft (quad) - spans x=[0, 0.5], y=±0.25
        vertices['a_template_pos'][0] = [0.0, -0.25]  # shaft bottom-left
        vertices['a_template_pos'][1] = [0.0, 0.25]  # shaft top-left
        vertices['a_template_pos'][2] = [0.5, -0.25]  # shaft bottom-right
        vertices['a_template_pos'][3] = [0.5, 0.25]  # shaft top-right

        # Head (triangle) - spans x=[0.5, 1.0], y=±0.75 at base, 0 at tip
        vertices['a_template_pos'][4] = [0.5, -0.75]  # head bottom base
        vertices['a_template_pos'][5] = [0.5, 0.75]  # head top base
        vertices['a_template_pos'][6] = [1.0, 0.0]  # head tip

        # Indices: 2 triangles for shaft + 1 triangle for head
        indices = np.array(
            [
                0,
                1,
                2,  # shaft triangle 1
                1,
                2,
                3,  # shaft triangle 2
                4,
                5,
                6,  # head triangle
            ],
            dtype=np.uint32,
        )

        self._template_vbo.set_data(vertices)
        self._index_buffer.set_data(indices)
        self._template_vertex_count = 7
        self._template_index_count = 9

    def set_data(
        self,
        vertices=None,
        faces=None,
        face_colors=None,
        vector_style='line',
    ):
        """Set data for rendering vectors.

        Parameters
        ----------
        vertices : (N, D) array
            Vertices in pairs (start, end) for each line segment
        faces : array
            Ignored (for compatibility with mesh API)
        face_colors : (N, 4) array
            Colors for each vertex
        vector_style : str
            One of 'line', 'triangle', or 'arrow'
        """
        # Store style (will be used in shader)
        self._vector_style = vector_style

        if vertices is None or len(vertices) == 0:
            self._data = None
            self._n_instances = 0
            return

        # Vertices come in as consecutive pairs: [start0, end0, start1, end1, ...]
        assert len(vertices) % 2 == 0, 'Vertices must be pairs (start, end)'

        ndim = vertices.shape[1]
        n_segments = len(vertices) // 2

        # Create instance data using structured array for vispy compatibility
        instance_data = np.zeros(
            n_segments,
            dtype=[
                ('a_start', np.float32, 3),
                ('a_end', np.float32, 3),
                ('a_color', np.float32, 4),
            ],
        )

        # Fill using vectorized operations - vertices are [start0, end0, start1, end1, ...]
        instance_data['a_start'][:, :ndim] = vertices[
            ::2
        ]  # All starts (even indices)
        instance_data['a_end'][:, :ndim] = vertices[
            1::2
        ]  # All ends (odd indices)

        # One color per instance (per vector)
        if face_colors is not None and len(face_colors) > 0:
            assert len(face_colors) == n_segments, (
                'face_colors must have one color per vector'
            )
            instance_data['a_color'] = face_colors
        else:
            # Default white color
            instance_data['a_color'] = [1, 1, 1, 1]

        self._data = instance_data
        self._n_instances = n_segments

        # Set up vertex buffers
        self._instance_vbo.set_data(instance_data)

        # Always bind template vertices (they're small and rebinding is cheap)
        a_template_pos = self._template_vbo['a_template_pos']
        self.shared_program['a_template_pos'] = a_template_pos

        # Always bind instance attributes (VBO was just updated)
        a_start = self._instance_vbo['a_start']
        a_start.divisor = 1
        self.shared_program['a_start'] = a_start

        a_end = self._instance_vbo['a_end']
        a_end.divisor = 1
        self.shared_program['a_end'] = a_end

        a_color = self._instance_vbo['a_color']
        a_color.divisor = 1
        self.shared_program['a_color'] = a_color

        # Set draw mode and index buffer
        self._vshare.draw_mode = 'triangles'
        self._vshare.index_buffer = self._index_buffer

        self.update()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        self.shared_program['u_width'] = value
        self.update()

    def _prepare_transforms(self, view):
        view.view_program.vert['visual_to_framebuffer'] = view.get_transform(
            'visual', 'framebuffer'
        )
        view.view_program.vert['framebuffer_to_render'] = view.get_transform(
            'framebuffer', 'render'
        )

    def _prepare_draw(self, view):
        if self._data is None or self._n_instances == 0:
            return False

        # Set uniforms
        view.view_program['u_px_scale'] = view.transforms.pixel_scale
        view.view_program['u_width'] = self._width

        # Head length/width ratios create the line/triangle/arrow appearance.
        # The triangle base and the line both have width == edge_width (as in
        # the original mesh renderer), so the triangle uses head_width_ratio
        # = 1.0 rather than the arrow's wider head. (For the line the head is
        # a degenerate zero-area triangle, so its width ratio is irrelevant.)
        if self._vector_style == 'line':
            # All shaft, no head
            view.view_program['u_head_length_ratio'] = 0.0
            view.view_program['u_head_width_ratio'] = 1.0
        elif self._vector_style == 'triangle':
            # All head, no shaft; base width matches a line of the same width
            view.view_program['u_head_length_ratio'] = 1.0
            view.view_program['u_head_width_ratio'] = 1.0
        else:  # arrow: shaft + wider head (hardcoded ratio for now)
            view.view_program['u_head_length_ratio'] = 0.25
            view.view_program['u_head_width_ratio'] = 4.0

        # Calculate direction-independent data-to-screen scale for data mode
        # Transform unit basis vectors and average their screen-space lengths
        tr = view.get_transform('visual', 'framebuffer')
        # Sample a point (use origin)
        origin = np.array([[0, 0, 0, 1]], dtype=np.float32)
        basis_x = np.array([[1, 0, 0, 1]], dtype=np.float32)
        basis_y = np.array([[0, 1, 0, 1]], dtype=np.float32)
        basis_z = np.array([[0, 0, 1, 1]], dtype=np.float32)

        origin_fb = tr.map(origin)[0]
        x_fb = tr.map(basis_x)[0]
        y_fb = tr.map(basis_y)[0]
        z_fb = tr.map(basis_z)[0]

        # Perspective divide and measure lengths
        origin_fb = origin_fb[:2] / origin_fb[3]
        x_fb = x_fb[:2] / x_fb[3]
        y_fb = y_fb[:2] / y_fb[3]
        z_fb = z_fb[:2] / z_fb[3]

        scale_x = np.linalg.norm(x_fb - origin_fb)
        scale_y = np.linalg.norm(y_fb - origin_fb)
        scale_z = np.linalg.norm(z_fb - origin_fb)

        # Average scale across all axes
        data_scale = (scale_x + scale_y + scale_z) / 3.0
        view.view_program['u_data_scale'] = float(data_scale)

        return True

    # TODO: bounds are not reported correctly at the moment! We should
    # overload `_compute_bounds` based on the instances


Vectors = create_visual_node(VectorsVisual)
